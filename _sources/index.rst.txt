SUSPECT: Special Structure Detection for Pyomo
==============================================

Release v\ |version|.

.. note:: SUSPECT requires Python 3. If you are using Python 2, please
          upgrade.


.. _installation:


Installation
------------

You can get SUSPECT from Pypi_::

  pip install cog-suspect

.. _Pypi: https://pypi.org/project/cog-suspect/


Quick Start
-----------


After we have constructed the model, we can pass it to
:py:func:`suspect.detect_special_structure` to obtain special
structure information

.. code-block:: python

    from suspect import detect_special_structure
    import pyomo.environ as aml


    model = aml.ConcreteModel()
    model.x = aml.Var()
    model.y = aml.Var()

    model.obj = aml.Objective(expr=(model.y - model.x)**3)
    model.c1 = aml.Constraint(expr=model.y - model.x >= 0)

    info = detect_special_structure(model)

    # try info.variables, info.objectives, and info.constraints
    # in this case, objective is not detected as convex
    print(info.objectives['obj'])


We can convert the Pyomo model to a _connected_ Pyomo model, where common sub-expressions
are connected together in a Directed Acyclic Graph (DAG). With this extra information,
SUSPECT will detect the objective as convex.

.. code-block:: python

    from suspect import create_connected_model

    connected, _ = create_connected_model(model)

    info = detect_special_structure(connected)

    # now the objective is detected as convex!
    print(info.objectives['obj'])


Quadratic Expression Support
----------------------------

SUSPECT extends Pyomo to include Quadratic expressions.
If you use this feature, you should need to call the following function at the beginning of
your script:

.. code-block:: python

    from suspect.pyomo import enable_standard_repn_for_quadratic_expression

    enable_standard_repn_for_quadratic_expression()


Command Line Usage
------------------

SUSPECT comes with a command line tool to quickly inspect an
optimization problem in the OSiL format::

    $ model_summary.py -p /path/to/problem/instance.osil -s /path/to/problem/solution.p1.sol

This command will print a summary about the problem objective and constraints, for example::

    $ model_summary.py -p /path/to/rsyn0805h.osil -s /path/to/rsyn0805h.p1.sol
    INFO:root:	Reading Problem
    INFO:root:	Converting DAG
    INFO:root:	Starting Special Structure Detection
    INFO:root:	Special Structure Detection Finished
    {"bounds_obj_ok": true, "bounds_var_ok": true, "conscurvature": "convex", "name": "rsyn0805h", "nbinvars": 37, "ncons": 429, "nintvars": 0, "nvars": 308, "objcurvature": "linear", "objsense": "max", "objtype": "linear", "runtime": 0.21518850326538086, "status": "ok"}


API Documentation
-----------------

.. toctree::
   :maxdepth: 2

   api
