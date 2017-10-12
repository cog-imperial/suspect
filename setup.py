from setuptools import setup, find_packages

setup(
    name='convexity_detection',
    author='Francesco Ceccon',
    author_email='francesco@ceccon.me',
    packages=find_packages('convexity_detection'),
    requires=['pyomo', 'numpy', 'mpmath'],
    setup_requires=['pytest-runner'],
    tests_requires=['pytest', 'pytest-cov'],
)
