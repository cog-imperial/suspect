Special Structure Detection for Pyomo
=====================================

This library implements methods to:

* Detect convex and concave expressions
* Detect increasing and decreasing expressions
* Detect linear, quadratic and polynomial expressions
* Tighten expression bounds


Installation
------------

Install by running `python setup.py install`.


Usage
-----

The package contains an utility to display structure information about
a single problem.

You can run the utility as:

    model_summary.py -p /path/to/problem.osil

or, if you want to check variables bounds include the solution:

    model_summary.py -p /path/to/problem.osil -s /path/to/problem.sol

The repository also includes a `Dockerfile` to simplify running the utility in
batch mode in a cloud environment. Refer to the `batch` folder for more information.


References
----------

 * R Fourer, D Orban. DrAmpl: A meta solver for optimization problem analysis. Computational Management Science. 2010
 * R Fourer et al. Convexity and Concavity Detection in Computational Graphs: Tree Walks for Convexity Assessment. INFORMS Journal on Computing. 2010
