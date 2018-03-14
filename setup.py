from setuptools import setup, find_packages
from pathlib import Path

project_root = Path(__file__).resolve().parent

about = {}
version_path = project_root / 'suspect' / '__version__.py'
with version_path.open() as f:
    exec(f.read(), about)


setup(
    name='suspect',
    author=about['__author__'],
    author_email=about['__author_email__'],
    license=about['__license__'],
    version=about['__version__'],
    packages=find_packages(exclude=['tests']),
    entry_points={
        'suspect.convexity_detection': [
            'rsyn=suspect.extras.convexity:RSynConvexityVisitor',
            'l2norm=suspect.extras.convexity:L2NormConvexityVisitor',
            'quadratic=suspect.extras.convexity:QuadraticFormConvexityVisitor',
        ]
    },
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
