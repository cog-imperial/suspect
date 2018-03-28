Extending SUSPECT
=================


Tutorial
--------

Pre requisites
~~~~~~~~~~~~~~

We recommend developing SUSPECT extensions in a separate virtual
environment than your main one.

From the root folder of your project, run::

  $ python3 -m venv myenv
  $ source myenv/bin/activate

After that, proceed to install SUSPECT in the newly created virtual
environment, see :ref:`installation`.


Task Overview
~~~~~~~~~~~~~

The objective of this tutorial is writing an extension for SUSPECT to detect
the convexity of the expression

.. math::

   f(x) = \frac{1}{x}

When :math:`x > 0`, :math:`f(x)` is convex, while when :math:`x < 0`,
:math:`f(x)` is concave.

In SUSPECT, this expression is represented by a root vertex of type
:class:`suspect.dag.expressions.DivisionExpression` with two children: a
:class:`suspect.dag.expressions.Constant` numerator and a
:class:`suspect.dag.expressions.Variable` denominator.

In this tutorial we will use the `procsyn <http://www.minlplib.org/procsyn.html>`_ instance
as an example.

At the moment, SUSPECT fails at detecting convexity of the constraints
containing the expression :math:`1/x`::

  $ model_summary.py -p /path/to/minlplib2/data/osil/procsyn.osil
  INFO:botocore.credentials:Found credentials in shared credentials file: ~/.aws/credentials
  INFO:root:Starting ../minlplib2/data/osil/procsyn.osil / None. timeout=300
  INFO:root:	Reading Problem
  INFO:root:	Converting DAG
  INFO:root:	Starting Special Structure Detection
  INFO:root:	Special Structure Detection Finished
  {"conscurvature": "indefinite", "name": "procsyn", "nbinvars": 0, "ncons": 27, "nintvars": 0,
   "nvars": 20, "objcurvature": "convex", "objsense": "min", "objtype": "polynomial",
   "runtime": 0.020023822784423828, "status": "ok"}



Detector Implementation
~~~~~~~~~~~~~~~~~~~~~~~

At the root of your project directory, create a new directory
containing the source of the convexity detector::

  $ mkdir example_ext

And write the detector in ``__init__.py``::

  $ $EDITOR example_ext/__init__.py

In this way we can later on package the detector. The content of the file is as follows

.. code-block:: python
   :emphasize-lines: 42

    from suspect.ext import ConvexityDetector, Convexity
    import suspect.dag.expressions as dex


    class ConstOverVarDetector(ConvexityDetector):
	"""Convexity detector for expressions of the type:

	.. math::

	    \frac{c}{x}

	where c is a constant and x is a variable.
	"""

	def register_handlers(self):
	    """Register interest in DivisionExpression."""
	    return {
		dex.DivisionExpression: self.visit_division,
	    }

	def visit_division(self, expr, ctx):
	    """Visit a division expression.

	    Return `Convexity.Convex` if the numerator is constant and the
	    denominator is nonnegative.

	    Return `Convexity.Concave` if the numerator is constant and the
	    denominator is nonpositive.
	    """
	    assert len(expr.children) == 2
	    num, den = expr.children

	    if not isinstance(num, dex.Constant):
		return

	    if not isinstance(den, dex.Variable):
		return

	    if num.value == 0:
		return Convexity.Linear

	    bound = ctx.bound[den]

	    if bound.is_nonnegative():
		cvx = Convexity.Convex
	    elif bound.is_nonpositive():
		cvx = Convexity.Concave
	    else:
		return

	    if num.value > 0:
		return cvx
	    else:
		return cvx.negate()


Implementing a convexity detector requires subclassing the
``ConvexityDetector`` base class and implement ``register_handlers``
to tell SUSPECT what kind of expressions our detector can handle.
In our case, we only handle divisions.

In the ``visit_divison`` callback we first check if the numerator is a
constant and the denominator a variable.

In the highlighted line, we lookup the bound for the denominator, if
this bound is nonnegative then the expression is convex, otherwise it
is concave. We finally handle the case when the constant is negative,
and in that case we return the negation of our computed convexity
information.



Packaging
~~~~~~~~~

SUSPECT requires extensions to be packaged and registered as an entry
point. At the root of your project, add the following ``setup.py`` file


.. code-block:: python
   :emphasize-lines: 7,8,9,10,11

    from setuptools import setup


    setup(
	name='example_ext',
	packages=['example_ext'],
	entry_points={
	    'suspect.convexity_detection': [
		'const_over_var=example_ext:ConstOverVarDetector'
	    ]
	},
	requires=['suspect']
    )


The highlighted lines show how to register the convexity detector with SUSPECT.

Finally, install the convexity detector::

  $ python setup.py install


If we now run SUSPECT with the same input file as before, we can see
that the convexity of the constraints is correctly identified::

  $ model_summary.py -p /path/to/minlplib2/data/osil/procsyn.osil
  INFO:botocore.credentials:Found credentials in shared credentials file: ~/.aws/credentials
  INFO:root:Starting ../minlplib2/data/osil/procsyn.osil / None. timeout=300
  INFO:root:	Reading Problem
  INFO:root:	Converting DAG
  INFO:root:	Starting Special Structure Detection
  INFO:root:	Special Structure Detection Finished
  {"conscurvature": "convex", "name": "procsyn", "nbinvars": 0, "ncons": 27, "nintvars": 0,
   "nvars": 20, "objcurvature": "convex", "objsense": "min", "objtype": "polynomial",
   "runtime": 0.01907968521118164, "status": "ok"}
