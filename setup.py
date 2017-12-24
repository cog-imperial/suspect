from setuptools import setup, find_packages

setup(
    name='suspect',
    author='Francesco Ceccon',
    author_email='francesco@ceccon.me',
    packages=[
        'suspect',
        'suspect.monotonicity',
        'suspect.convexity'
    ],
    scripts=[
        'scripts/model_summary.py'
    ],
    requires=['pyomo', 'numpy', 'mpmath'],
    setup_requires=['pytest-runner'],
    tests_require=['pytest', 'pytest-cov'],
)
