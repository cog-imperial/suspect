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
from io import StringIO


_NODE_SYMBOLS = {
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


def _node_name(v):
    return 'n{}'.format(id(v))


def _node_label(expr):
    if isinstance(expr, (dex.Variable, dex.Constraint, dex.Objective)):
        return expr.name
    elif isinstance(expr, dex.Constant):
        return str(expr.lower_bound)
    else:
        return _NODE_SYMBOLS.get(type(expr))


def dump(dag, f, metadata=None):
    f.write('digraph G {\n')
    f.write('  rankdir = TB;\n')
    f.write('  rank = same;\n')

    f.write('  { rank=same;\n')
    for cons in dag.constraints.values():
        f.write('    {} [label="{}", shape=box];\n'.format(
            _node_name(cons),
            _node_label(cons),
        ))

    for obj in dag.objectives.values():
        f.write('    {} [label="{}", shape=box, color=red];\n'.format(
            _node_name(obj),
            _node_label(obj),
        ))
    f.write('  }\n')

    cur_depth = 0
    f.write('  { rank=same;\n')
    for vertex in dag.vertices:
        if isinstance(vertex, (dex.Constraint, dex.Objective)):
            continue
        if vertex.depth != cur_depth:
            cur_depth += 1
            f.write('  }\n  { rankdir=same;\n')

        if metadata is None or metadata.get(vertex) is None:
            label = _node_label(vertex)
        elif metadata is not None and metadata.get(vertex) is not None:
            label = _node_label(vertex) + '\\n' + str(metadata[vertex])
        f.write('    {} [label="{}"];\n'.format(
            _node_name(vertex),
            label,
        ))
    f.write('  }\n')

    for vertex in dag.vertices:
        for i, child in enumerate(vertex.children):
            f.write('  {} -> {} [taillabel="{}"];\n'.format(
                _node_name(vertex),
                _node_name(child),
                i,
            ))

    f.write('}\n')


def dumps(dag, metadata=None):
    s = StringIO()
    dump(dag, s, metadata)
    return s.getvalue()
