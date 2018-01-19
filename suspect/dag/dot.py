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

import suspect.dag.expressions as dex


NODE_SYMBOLS = {
    dex.ProductExpression: '*',
    dex.DivisionExpression: '/',
    dex.SumExpression: '+',
    dex.PowExpression: '^',
    dex.LinearExpression: 'Σ',
    dex.NegationExpression: '-',
    dex.AbsExpression: '|.|',
    dex.SqrtExpression: '√',
    dex.ExpExpression: 'exp',
    dex.LogExpression: 'log',
    dex.SinExpression: 'sin',
    dex.CosExpression: 'cos',
    dex.TanExpression: 'tan',
    dex.AsinExpression: 'asin',
    dex.AcosExpression: 'acos',
    dex.AtanExpression: 'atan',
}


def node_name(v):
    return 'n{}'.format(id(v))


def node_label(expr):
    if isinstance(expr, (dex.Variable, dex.Constraint, dex.Objective)):
        return expr.name
    elif isinstance(expr, dex.Constant):
        return str(expr.lower_bound)
    else:
        return NODE_SYMBOLS.get(type(expr))


def dump(dag, f):
    f.write('digraph G {\n')
    f.write('  rankdir = TB;\n')
    f.write('  rank = same;\n')

    f.write('  { rank=same;\n')
    for cons in dag.constraints.values():
        f.write('    {} [label="{}", shape=box];\n'.format(
            node_name(cons),
            node_label(cons),
        ))

    for obj in dag.objectives.values():
        f.write('    {} [label="{}", shape=box, color=red];\n'.format(
            node_name(obj),
            node_label(obj),
        ))
    f.write('  }\n')

    cur_depth = 0
    f.write('  { rank=same;\n')
    for vertex in dag.vertices:
        if isinstance(vertex, (dex.Constraint, dex.Objective)):
            continue
        if vertex.depth != cur_depth:
            cur_depth += 1
            f.write('  }{ rankdir=same;\n')

        f.write('  {} [label="{}"];\n'.format(
            node_name(vertex),
            node_label(vertex),
        ))
    f.write('}\n')

    for vertex in dag.vertices:
        for child in vertex.children:
            f.write('  {} -> {};\n'.format(
                node_name(vertex),
                node_name(child),
            ))

    f.write('}\n')
