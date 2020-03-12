#!/usr/bin/env python

# Copyright 2017 Francesco Ceccon
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import argparse
import os
import signal
import sys
import time
import logging
import json
import tempfile
import traceback
from mpmath import mpf
import boto3
from suspect.pyomo import create_connected_model, read_osil, enable_standard_repn_for_quadratic_expression
from suspect import (
    detect_special_structure,
    logger,
)

enable_standard_repn_for_quadratic_expression()


class Timeout(object):
    def __init__(self, seconds=1, error_message='Timeout'):
        self.seconds = seconds
        self.error_message = error_message

    def handle_timeout(self, signum, frame):
        raise TimeoutError(self.error_message)

    def __enter__(self):
        signal.signal(signal.SIGALRM, self.handle_timeout)
        signal.alarm(self.seconds)

    def __exit__(self, type, value, traceback):
        signal.alarm(0)


class S3Storage(object):
    PREFIX = 's3://'

    def __init__(self):
        self.s3 = boto3.resource('s3')

    def download_to_temp(self, resource, prefix=None):
        bucket = self._bucket(resource)
        path = self._path(resource)
        temp_dir = tempfile.TemporaryDirectory(prefix=prefix)
        filename = os.path.basename(path)
        dest_file = os.path.join(temp_dir.name, filename)
        bucket.download_file(path, dest_file)
        logging.info('Downloaded {} to {}'.format(resource, dest_file))
        return dest_file, temp_dir

    def upload_file(self, orig, dest):
        bucket = self._bucket(dest)
        path = self._path(dest)
        logging.info('Uploading {} to {}'.format(orig, dest))
        bucket.upload_file(orig, path)

    def _bucket(self, resource):
        resource = resource[len(self.PREFIX):]
        parts = resource.split('/')
        return self.s3.Bucket(parts[0])

    def _path(self, resource):
        resource = resource[len(self.PREFIX):]
        parts = resource.split('/')
        return '/'.join(parts[1:])


class RunResources(object):
    """Context Manager to provide uniform interface for local and S3 objects"""

    def __init__(self, problem, solution, output):
        s3 = S3Storage()
        self.s3 = s3

        if problem.startswith(s3.PREFIX):
            # keep a ref to problem dir so it does not get cleaned self
            self.problem, self.problem_dir = \
                s3.download_to_temp(problem, prefix='problem')
        else:
            self.problem = problem

        if solution is not None and solution.startswith(s3.PREFIX):
            # keep a ref to solution dir so it does not get cleaned self
            self.solution, self.solution_dir = \
                s3.download_to_temp(solution, prefix='solution')
        else:
            self.solution = solution

        self.output_type = None
        if output is None:
            self.output_owned = False
            self.output = sys.stdout
        else:
            self.output_owned = True
            if output.startswith(s3.PREFIX):
                self.output = tempfile.NamedTemporaryFile('w', prefix='output')
                self.output_dest = output
                self.output_type = 's3'
            else:
                self.output = open(output, 'w')
                self.output_type = 'fs'

    def __enter__(self):
        return self

    def __exit__(self, type, value, traceback):
        if self.output_type == 'fs':
            self.output.close()
        elif self.output_type == 's3':
            self.output.flush()
            self.s3.upload_file(self.output.name, self.output_dest)


def read_problem(filename):
    if filename.endswith('.osil'):
        model = read_osil(
            filename,
            objective_prefix='obj_',
            constraint_prefix='cons_',
        )

        if model.name is None:
            basename = os.path.basename(filename)
            model.name = os.path.splitext(basename)[0]

        return model
    else:
        raise RuntimeError('Unknown file type.')


def _conscurvature(info):
    curvatures = info.conscurvature().values()
    if all(c.is_linear() for c in curvatures):
        return 'linear'
    elif all(c.is_convex() for c in curvatures):
        return 'convex'
    elif all(c.is_concave() for c in curvatures):
        return 'concave'
    else:
        return 'indefinite'


def _objsense(info):
    senses = [o['sense'] for o in info.objectives.values()]
    if len(info.objectives) == 1:
        return senses[0]
    else:
        return ', '.join(senses)


def _objcurvature(info):
    curvatures = [c for c in info.objcurvature().values()]
    if len(info.objectives) == 1:
        cvx = curvatures[0]
        if cvx.is_linear():
            return 'linear'
        elif cvx.is_convex():
            return 'convex'
        elif cvx.is_concave():
            return 'concave'
        else:
            return 'indefinite'
    else:
        return ', '.join(curvatures)


def _objtype(info):
    objtypes = info.objtype().values()
    if len(objtypes) == 1:
        return list(objtypes)[0]
    else:
        return ', '.join(objtypes)


def summarize_problem_structure(info):
    return {
        'name': info.name,
        'nvars': info.num_variables(),
        'nbinvars': info.num_binaries(),
        'nintvars': info.num_integers(),
        'ncons': info.num_constraints(),
        'conscurvature': _conscurvature(info),
        'objsense': _objsense(info),
        'objcurvature': _objcurvature(info),
        'objtype': _objtype(info),
    }


def run_for_problem(filename, solution_filename, timeout_):
    # If we can't read a problem in, don't do anything.
    logging.info('Starting {} / {}. timeout={}'.format(filename, solution_filename, timeout_))
    try:
        with Timeout(seconds=300):
            logging.info('\tReading Problem')
            model = read_problem(filename)
            logging.info('\tConverting DAG')
            model, _ = create_connected_model(model)
    except Exception as err:
        logging.error('{}: Error reading: {}'.format(filename, str(err)))
        return None

    try:
        with Timeout(seconds=timeout_):
            logging.info('\tStarting Special Structure Detection')
            start_t = time.time()
            info = detect_special_structure(model)
            end_t = time.time()
            logging.info('\tSpecial Structure Detection Finished')

    except Exception as err:
        logging.error('{}: {}'.format(model.name, str(err)))
        traceback.print_exc()
        return {
            'name': model.name,
            'status': 'error',
            'error': str(err),
        }

    summary = summarize_problem_structure(info)
    summary['runtime'] = end_t - start_t
    summary['status'] = 'ok'

    if solution_filename is not None:
        try:
            var_bounds_ok, obj_bounds_ok = check_solution_bounds(
                info.variables, info.objectives, solution_filename
            )
            summary['bounds_var_ok'] = var_bounds_ok
            summary['bounds_obj_ok'] = obj_bounds_ok
        except Exception as err:
            logging.error('{}: Solution bounds: {}'.format(model.name, str(err)))

    return summary


def check_solution_bounds(variables, objectives, solution_filename):
    wrong_bounds = []
    objvar_value = None
    with open(solution_filename) as f:
        for line in f:
            parts = line.split()
            var_name = parts[0]
            sol = mpf(parts[1])

            if var_name == 'objvar':
                objvar_value = sol
            if var_name not in variables:
                continue

            v = variables[var_name]

            if sol < v['lower_bound'] or sol > v['upper_bound']:
                wrong_bounds.append(var_name)

    var_bounds_ok = len(wrong_bounds) == 0
    if len(objectives) != 1 or objvar_value is None:
        return var_bounds_ok, None

    for obj in objectives.values():
        lower_bound = obj['lower_bound']
        upper_bound = obj['upper_bound']
        obj_bounds_ok = (
                objvar_value >= lower_bound and objvar_value <= upper_bound
        )
        return var_bounds_ok, obj_bounds_ok


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)

    parser = argparse.ArgumentParser()
    parser.add_argument('--problem', '-p')
    parser.add_argument('--solution', '-s')
    parser.add_argument('--output', '-o', nargs='?')
    parser.add_argument('--timeout', type=int, default=300,
                        help='Timeout in seconds')
    parser.add_argument('--log', dest='log_level',
                        choices=['DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL'],
                        default='WARNING')
    args = parser.parse_args()

    logger.setLevel(level=args.log_level)

    if args.problem is None:
        parser.print_help()
        sys.exit(1)

    with RunResources(args.problem, args.solution, args.output) as r:
        result = run_for_problem(r.problem, r.solution, args.timeout)

        if result is None:
            sys.exit(1)

        result_str = json.dumps(result, sort_keys=True)
        r.output.write(result_str + '\n')
