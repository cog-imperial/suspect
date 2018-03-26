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
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib2tikz import save as tikz_save


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


def print_runtime_stats(suspect):
    data = suspect.runtime
    bins = [0.0, 1.0, 2.0, 5.0, 10.0, 60.0, 120.0, data.max()]
    binned = pd.cut(data, bins, retbins=False)
    hist = binned.groupby(binned).count()
    tot = hist.sum()
    max_width = 30
    print('Runtime:')
    print()
    for n, c in hist.items():
        fract = c / tot
        bar = '*' * int(max_width * fract)
        print('  ({:-6.2f}, {:-6.2f}] {:4d} {:6.2f}% |{}'.format(
            n.left, n.right, c, fract * 100, bar
        ))
    print()


def plot_runtime_stats(suspect, total_instances, read_instances=None):
    data = suspect.runtime.sort_values()
    data_len = data.shape[0]
    pct_done = np.arange(data_len) / total_instances
    pct_data = pd.Series(pct_done, index=data.values)
    fig = plt.figure(figsize=(8, 4))
    pct_data.plot(logx=True, fig=fig)
    if read_instances is not None:
        read_pct = read_instances / total_instances
        plt.axhline(read_pct, color='black', linestyle='dashed', linewidth=0.8)
    plt.xlabel('Time ($\log s$)')
    plt.ylabel('Instances Processed ($\%$)')
    plt.ylim([0, 1])
    plt.xlim([pct_data.index.min(), pct_data.index.max()])
    plt.grid()
    return fig


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


def write_problem_list(filename, df, columns=3):
    with open(filename, 'w') as f:
        count = 0
        for name in df.index:
            f.write('  ' + str(name.replace('_', '\\_')) + '  ')
            if count >= (columns - 1):
                f.write('\\\\\n')
                count = 0
            else:
                f.write(' & ')
                count += 1


if __name__ == '__main__':
    args = argparse.ArgumentParser()
    args.add_argument('suspect_input')
    args.add_argument('minlplib_input')
    args = args.parse_args()

    suspect = pd.read_csv(args.suspect_input)
    minlplib = pd.read_csv(args.minlplib_input)

    total_instances = minlplib.shape[0]

    print()
    print('=' * 20 + '-- INPUT DATA --' + '=' * 20)
    print()

    print('MINLPLib 2: {} instances'.format(minlplib.shape[0]))
    print()


    print_error_statistics(suspect)

    suspect_orig = suspect.set_index('name')

    # filter out errors
    suspect = suspect[suspect.status == 'ok']

    good_instances = check_input_data_correctness(suspect, minlplib)

    suspect.set_index('name', inplace=True)
    minlplib.set_index('name', inplace=True)

    missing_problems = minlplib.loc[~minlplib.index.isin(suspect_orig.index)]

    print()
    print('=' * 21 + '-- RESULTS --' + '=' * 21)
    print()
    print('Correctly read {} out of {} problems ({:.2f}%)'.format(
        suspect_orig.shape[0], total_instances, (suspect_orig.shape[0] / total_instances) * 100
    ))

    print_runtime_stats(suspect.loc[good_instances])
    runtime_plot = plot_runtime_stats(suspect.loc[good_instances], total_instances, suspect_orig.shape[0])

    suspect = suspect.loc[good_instances, STRUCTURE_COLUMNS]
    minlplib = minlplib.loc[good_instances, STRUCTURE_COLUMNS]

    # suspect.drop('gams03', inplace=True)
    # minlplib.drop('gams03', inplace=True)

    cvx_replace = {
        'unknown': 'indefinite',
        'nonconcave': 'indefinite',
        'nonconvex': 'indefinite',
    }
    minlplib.conscurvature.replace(cvx_replace, inplace=True)
    minlplib.objcurvature.replace(cvx_replace, inplace=True)

    type_replace = {
        'signomial': 'nonlinear'
    }
    minlplib.objtype.replace(type_replace, inplace=True)

    diff = suspect == minlplib
    all_good = diff.all(axis=1)
    good_pct = sum(all_good) / all_good.shape[0]

    print('Correctly identified {} out of {} ({:.2f}%) of the problems.'.format(
        sum(all_good),
        all_good.shape[0],
        good_pct*100
    ))
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

    good_idx = all_good[all_good].index
    wrong_idx = all_good[~all_good].index
    correct_result = suspect_orig.loc[good_idx]
    wrong_result = suspect_orig.loc[wrong_idx]

    correct_lt2 = correct_result[correct_result.runtime <= 2]
    correct_lt10 = correct_result[(correct_result.runtime > 2) & (correct_result.runtime <= 10)]
    correct_gt10 = correct_result[correct_result.runtime > 10]

    compare_wrong = wrong_result[STRUCTURE_COLUMNS].join(minlplib[STRUCTURE_COLUMNS], how='left', lsuffix='_have', rsuffix='_expected')
