Changelog
=========

2.1.2 (2021-02-16)
------------------

* Make SUSPECT compatible with Pyomo 5.7.2+

2.1.1 (2020-11-13)
------------------

* Try to compute special structure even if bounds are missing

2.1.0 (2020-09-16)
------------------

* Add support for `log10`
* Improve handling of quadratic expressions in nonlinear problems
* Fix FBBT bug when handling some types of expressions

2.0.2 (2020-09-01)
------------------

* Fix convexity on division
* Handle Pyomo `MonomialTermExpression`

2.0.1 (2020-07-01)
------------------

* Minor bug fixes

2.0.0 (2020-04-28)
------------------

* Use Pyomo expressions to represent the DAG
* Replace DAG with connected_model

1.6.0 (2019-11-15)
------------------

* Add floating point math mode
* Minor performance fixes

1.1.0 (2019-01-31)
------------------

* Add Quadratic expression type
* Add Interval special case for x*x
* Fix Interval sin
* Add Interval comparison with numbers

1.0.7 (2018-08-30)
------------------

* Add Interval abs
* Add Interval power


1.0.6 (2018-07-05)
------------------

* Change ExpressionType and UnaryFunctionType to IntEnum


1.0.5 (2018-07-05)
------------------

* Documentation improvements


1.0.4 (2018-07-04)
------------------

* First public release. Yay!
