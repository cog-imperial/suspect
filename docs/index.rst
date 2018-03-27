SUSPECT: Special Structure Detection for Pyomo
==============================================

Release v\ |version|.

.. note:: SUSPECT requires Python 3. If you are using Python 2, please
          upgrade.


.. _installation:

Installation
------------

At the moment, SUSPECT is not available as a Python package and you need to manually install it.

Clone the SUSPECT repository::

    $ git clone git://github.com/cog-imperial/suspect.git

Or download the tarball::

    $ curl -OL https://github.com/cog-imperial/suspect/tarball/master

After you downloaded the source, you can install it in the current environment::

    $ cd suspect
    $ pip install -r requirements.txt
    $ python install .



Quick Start
-----------

SUSPECT requires Pyomo 4 expression trees to work, so the first thing
to do is switch to this expression representation.

.. code-block:: python

   from suspect import set_pyomo4_expression_tree

   set_pyomo4_expression_tree()


After we have constructed the model, we can pass it to
:py:func:`suspect.detect_special_structure` to obtain special
structure information

.. code-block:: python

    from suspect import (
	set_pyomo4_expression_tree,
	detect_special_structure,
    )
    import pyomo.environ as aml


    set_pyomo4_expression_tree()


    model = aml.ConcreteModel()
    model.x = aml.Var()
    model.y = aml.Var()

    model.obj = aml.Objective(expr=(model.y - model.x)**3)
    model.c1 = aml.Constraint(expr=model.y - model.x >= 0)

    info = detect_special_structure(model)

    # try info.variables, info.objectives, and info.constraints
    print(info.objectives['obj'])



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


Extending SUSPECT
-----------------

.. toctree::
   :maxdepth: 2

   extending


API Documentation
-----------------

.. toctree::
   :maxdepth: 2

   api
