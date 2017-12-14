#!/bin/env python

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
from contextlib import contextmanager
from mpmath import mpf
from convexity_detection.osil_reader import read_osil
from convexity_detection import (
    set_pyomo4_expression_tree,
    detect_special_structure,
)


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


@contextmanager
def ScriptOutput(output):
    try:
        if output is None:
            owned = False
            output = sys.stdout
        else:
            owned = True
            output = open(output, 'w')

        yield output
    finally:
        if owned:
            output.close()


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
    curvatures = [str(c) for c in info.objcurvature().values()]
    if len(info.objectives) == 1:
        return curvatures[0]
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
            model = read_problem(filename)
    except Exception as err:
        logging.error('{}: Error reading: {}'.format(filename, str(err)))
        return None

    try:
        with Timeout(seconds=timeout_):
            start_t = time.time()
            info = detect_special_structure(model)
            end_t = time.time()

    except Exception as err:
        logging.error('{}: {}'.format(model.name, str(err)))
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
    set_pyomo4_expression_tree()
    logging.basicConfig(level=logging.INFO)

    parser = argparse.ArgumentParser()
    parser.add_argument('--problem', '-p')
    parser.add_argument('--solution', '-s')
    parser.add_argument('--output', '-o', nargs='?')
    parser.add_argument('--timeout', type=int, default=300,
                        help='Timeout in seconds')
    args = parser.parse_args()

    if args.problem is None:
        parser.print_help()
        sys.exit(1)

    result = run_for_problem(args.problem, args.solution, args.timeout)
    if result is None:
        sys.exit(1)

    result_str = json.dumps(result, sort_keys=True)
    with ScriptOutput(args.output) as output:
        output.write(result_str + '\n')
        output.flush()
