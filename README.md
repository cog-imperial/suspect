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
a single problem or, in batch mode, a list of problems.

You can run the utility as:

    model_summary.py -i /path/to/problem.osil

or, if you want to check variables bounds include the solution:

    model_summary.py -i /path/to/problem.osil:/path/to/problem.sol


If you want to run the utility on a large set of problem, you can use batch mode:

    model_summary.py --batch -i /path/to/file/list.txt -o /path/to/output.csv

Where the input file is like:

    /path/to/problem1.osil:/path/to/problem1.sol
    /path/to/problem2.osil:/path/to/probelm2.sol
    ...



References
----------

 * R Fourer, D Orban. DrAmpl: A meta solver for optimization problem analysis. Computational Management Science. 2010
 * R Fourer et al. Convexity and Concavity Detection in Computational Graphs: Tree Walks for Convexity Assessment. INFORMS Journal on Computing. 2010
