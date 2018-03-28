Developer Interface
===================

.. module:: suspect


Main Module
-----------

.. autofunction:: set_pyomo4_expression_tree
.. autofunction:: detect_special_structure
.. autoclass:: suspect.summary.ModelInformation
   :members:


Pyomo Compatibility
-------------------

.. autofunction:: suspect.pyomo.dag_from_pyomo_model
.. autofunction:: suspect.pyomo.read_osil


Directed Acyclic Graph
----------------------

.. autoclass:: suspect.dag.ProblemDag
   :members:

.. automodule:: suspect.dag.dot
   :members: dump, dumps


Expression Types
----------------

.. automodule:: suspect.dag.expressions
   :members:
