from setuptools import setup, find_packages

setup(
    name='suspect',
    author='Francesco Ceccon',
    author_email='francesco@ceccon.me',
    version='3',
    packages=find_packages(exclude=['tests']),
    scripts=[
        'scripts/model_summary.py',
        'scripts/osil_to_dot.py',
    ],
    classifiers=[
        'Development Status :: 4 - Beta',
        'Intended Audience :: Science/Research',
        'Programming Language :: Python :: 3.6',
        'Topic :: Scientific/Engineering :: Mathematics',
    ],
    requires=['pyomo', 'numpy', 'mpmath'],
    setup_requires=['pytest-runner'],
    tests_require=['pytest', 'pytest-cov', 'hypothesis'],
)
