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
from terminaltables import SingleTable
from mpmath import mpf
from pyomo_osil import read_osil
from convexity_detection import detect_special_structure


HEADERS = [
    'name', 'status', 'nvars', 'nbinvars', 'nintvars', 'ncons',
    'conscurvature', 'objsense', 'objcurvature', 'objtype',
    'wrong_bounds', 'obj_lower_bound', 'obj_upper_bound'
]


class timeout(object):
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


def parse_input(line):
    parts = line.strip().split(':')
    if len(parts) == 1:
        return parts[0], None
    elif len(parts) == 2:
        return parts[0], parts[1]
    else:
        raise RuntimeError((
            'Expected input line in the form: ' +
            'problem_file:solution_file'))


def read_batch_input(filename):
    with open(filename) as f:
        problems = [parse_input(line) for line in f]
    return sorted(problems, key=lambda x: x[0])


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
    try:
        model = read_problem(filename)
    except:
        return None

    try:
        with timeout(seconds=timeout_):
            start_t = time.time()
            info = detect_special_structure(model)
            end_t = time.time()

            summary = summarize_problem_structure(info)
            summary['runtime'] = end_t - start_t
            summary['status'] = 'ok'

            if solution_filename is not None:
                wrong_bounds = check_solution_bounds(
                    info.variables, solution_filename
                )
                summary['wrong_bounds'] = len(wrong_bounds)
                if len(info.objectives) == 1:
                    obj = list(info.objectives.values())[0]
                    summary['obj_lower_bound'] = obj['lower_bound']
                    summary['obj_upper_bound'] = obj['upper_bound']

    except:
        summary = {
            'name': model.name,
            'status': 'error',
        }

    return summary


def check_solution_bounds(variables, solution_filename):
    wrong_bounds = []
    with open(solution_filename) as f:
        for line in f:
            parts = line.split()
            var_name = parts[0]
            sol = mpf(parts[1])
            if var_name not in variables:
                continue
            v = variables[var_name]

            if sol < v['lower_bound'] or sol > v['upper_bound']:
                wrong_bounds.append(var_name)

    return wrong_bounds


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', '-i')
    parser.add_argument('--output', '-o', nargs='?')
    parser.add_argument('--batch', action='store_true')
    parser.add_argument('--timeout', type=int, default=300,
                        help='Timeout in seconds')
    args = parser.parse_args()

    if args.input is None:
        parser.print_help()
        sys.exit(1)

    if args.batch:
        # batch mode
        problems = read_batch_input(args.input)
        if args.output is None:
            output = sys.stdout
        else:
            output = open(args.output, 'w')

        output.write(','.join(HEADERS) + '\n')
        for problem_filename, solution_filename in problems:
            result = run_for_problem(
                problem_filename,
                solution_filename,
                args.timeout,
            )
            if result is None:
                continue
            out = [str(result.get(k)) for k in HEADERS]
            output.write(','.join(out) + '\n')
            output.flush()

        if args.output is not None:
            output.close()
    else:
        # single file mode
        filename, solution_name = parse_input(args.input)
        result = run_for_problem(filename, solution_name, args.timeout)
        table = SingleTable([(k, v) for k, v in result.items()])
        print(table.table)
