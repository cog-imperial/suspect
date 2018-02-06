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
import pandas as pd


CHECK_COLUMNS = ['name', 'nvars', 'nintvars', 'nbinvars', 'ncons', 'objsense']
STRUCTURE_COLUMNS = ['conscurvature', 'objcurvature', 'objtype']


def print_error_statistics(data):
    good_len = sum(data.status == 'ok')
    good_pct = good_len / data.shape[0]

    print(('Solved {good_len} instances out of {total}: ' +
          '{good_pct:.2f}%.').format(
              good_len=good_len,
              total=data.shape[0],
              good_pct=good_pct*100
    ))

    errors = data[['name', 'error']][data.status != 'ok']
    for err_type, errs in errors.groupby('error'):
        print('Error: {} ({})'.format(err_type, errs.shape[0]))
        print('    ' + ', '.join(errs.name))
    print()


def check_input_data_correctness(suspect, minlplib):
    suspect = suspect[CHECK_COLUMNS].set_index('name')
    minlplib = minlplib[CHECK_COLUMNS].set_index('name')

    minlplib = minlplib.loc[suspect.index]

    diff = ~(minlplib == suspect).all(axis=1)
    diff_minlplib = minlplib.loc[diff[diff].index]
    diff_suspect = suspect.loc[diff[diff].index]

    print('Found problems in {} instances:'.format(diff_minlplib.shape[0]))
    print('    ' + ', '.join(diff_suspect.index))
    print()
    print('Expected:')
    print(diff_minlplib)
    print()
    print('Have:')
    print(diff_suspect)
    print()

    return diff[~diff].index


def compute_results_table(suspect, minlplib):
    expected_values = minlplib.unique()
    results = []
    for value in expected_values:
        expected = minlplib[minlplib == value]
        have = suspect.loc[expected.index]
        result = have.groupby(have).count().to_dict()
        for n, g in result.items():
            results.append({'expected': value, 'have': n, 'count': g})

    results = pd.DataFrame.from_records(results)
    return results.pivot(index='expected', columns='have', values='count').fillna(0)


if __name__ == '__main__':
    args = argparse.ArgumentParser()
    args.add_argument('suspect_input')
    args.add_argument('minlplib_input')
    args = args.parse_args()

    suspect = pd.read_csv(args.suspect_input)
    minlplib = pd.read_csv(args.minlplib_input)

    print()
    print('=' * 20 + '-- INPUT DATA --' + '=' * 20)
    print()
    print_error_statistics(suspect)

    # filter out errors
    suspect = suspect[suspect.status == 'ok']

    good_instances = check_input_data_correctness(suspect, minlplib)

    suspect.set_index('name', inplace=True)
    minlplib.set_index('name', inplace=True)

    suspect = suspect.loc[good_instances, STRUCTURE_COLUMNS]
    minlplib = minlplib.loc[good_instances, STRUCTURE_COLUMNS]

    suspect.drop('gams03', inplace=True)
    minlplib.drop('gams03', inplace=True)

    minlplib.conscurvature.replace({'unknown': 'indefinite'}, inplace=True)
    minlplib.objcurvature.replace({'unknown': 'indefinite'}, inplace=True)

    diff = suspect == minlplib
    all_good = diff.all(axis=1)
    good_pct = sum(all_good) / all_good.shape[0]

    print()
    print('=' * 21 + '-- RESULTS --' + '=' * 21)
    print()

    print('Correctly identified {:.2f}% of the problems.'.format(good_pct*100))
    print()
    print('Detailed breakdown:')
    print()

    print(' * Constraint Curvature')
    print(compute_results_table(suspect.conscurvature, minlplib.conscurvature))
    print()

    print(' * Objective Curvature')
    print(compute_results_table(suspect.objcurvature, minlplib.objcurvature))
    print()

    print(' * Objective Type')
    print(compute_results_table(suspect.objtype, minlplib.objtype))
