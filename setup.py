from setuptools import setup, find_packages

setup(
    name='convexity_detection',
    author='Francesco Ceccon',
    author_email='francesco@ceccon.me',
    packages=[
        'convexity_detection',
        'convexity_detection.monotonicity',
        'convexity_detection.convexity'
    ],
    scripts=[
        'scripts/model_summary.py'
    ],
    requires=['pyomo', 'numpy', 'mpmath'],
    setup_requires=['pytest-runner'],
    tests_require=['pytest', 'pytest-cov'],
)
