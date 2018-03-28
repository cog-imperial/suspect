#!/tmp/myenv/bin/python

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

import sys
import os
from suspect.pyomo.osil_reader import read_osil
from suspect.pyomo.convert import dag_from_pyomo_model
import suspect.dag.dot as dot
from suspect import set_pyomo4_expression_tree


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


if __name__ == '__main__':
    set_pyomo4_expression_tree()

    if len(sys.argv) != 3:
        print('Usage: osil_to_dot INPUT OUTPUT')
        sys.exit(1)

    problem = sys.argv[1]
    output = sys.argv[2]

    model = read_problem(problem)
    dag = dag_from_pyomo_model(model)
    with open(output, 'w') as f:
        dot.dump(dag, f)
