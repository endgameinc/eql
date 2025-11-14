#! /usr/bin/env python

"""Perform setup of the package for build."""
import sys
import glob
import os
import re
import io

from setuptools import setup, Command


with io.open('eql/__init__.py', 'rt', encoding='utf8') as f:
    __version__ = re.search(r'__version__ = \'(.*?)\'', f.read()).group(1)

install_requires = [
    "lark>=1.3.1",
]

test_requires = [
    "pytest>=6.0.0",
    "pytest-cov>=2.4",
    "flake8>=3.8.0",
    "pep257==0.7.0",
    "coverage>=4.5.3",
    "flake8-docstrings>=1.5.0",
    "PyYAML",
    "toml~=0.10",
    "attrs==21.4.0"
]
etc_files = [os.path.relpath(fn, 'eql') for fn in glob.glob('eql/etc/*') if not fn.endswith('.py')]


class Lint(Command):
    """Wrapper for the standard linters."""

    description = 'Lint the code'
    user_options = []

    def initialize_options(self):
        """Initialize options."""

    def finalize_options(self):
        """Finalize options."""

    def run(self):
        """Run the flake8 linter."""
        import subprocess
        # Run flake8 via command line (respects setup.cfg configuration)
        result = subprocess.run(
            [sys.executable, '-m', 'flake8', 'eql', 'tests'],
            cwd=os.path.dirname(os.path.abspath(__file__))
        )
        sys.exit(result.returncode)


class Test(Command):
    """Use pytest (http://pytest.org/latest/) in place of the standard unittest library."""

    user_options = [("pytest-args=", "a", "Arguments to pass to pytest")]

    def initialize_options(self):
        """Initialize options."""
        self.pytest_args = []

    def finalize_options(self):
        """Finalize options."""
        pass

    def run(self):
        """Run pytest."""
        import pytest
        import eql
        eql.parser.full_tracebacks = True
        sys.exit(pytest.main(self.pytest_args))


setup(
    name='eql',
    version=__version__,
    description='Event Query Language',
    install_requires=install_requires,
    author='Endgame, Inc.',
    author_email='eql@endgame.com',
    license='AGPLv3',
    classifiers=[
        'Intended Audience :: Developers',
        'Intended Audience :: Information Technology',
        'Intended Audience :: Science/Research',
        'Intended Audience :: System Administrators',
        'Natural Language :: English',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.8',
        'Programming Language :: Python :: 3.9',
        'Programming Language :: Python :: 3.10',
        'Programming Language :: Python :: 3.11',
        'Programming Language :: Python :: 3.12',
        'Topic :: Database',
        'Topic :: Internet :: Log Analysis',
        'Topic :: Scientific/Engineering :: Information Analysis',
    ],
    url='https://eql.readthedocs.io',
    cmdclass={
        'lint': Lint,
        'test': Test
    },
    entry_points={
        'console_scripts': [
            'eql=eql.main:main',
        ],
        'pygments.lexers': [
            'eql=eql.highlighters:EqlLexer'
        ]
    },
    extras_require={
        'docs': [
            'sphinx',
            'sphinx_rtd_theme',
        ],
        'cli': [
            'pygments',
            'prompt_toolkit',
        ],
        'lint': test_requires,
        'test': test_requires,
        'loaders': [
            'pyyaml',
            'toml',
        ],
        'highlighters': [
            'pygments',
        ]
    },
    packages=['eql', 'eql.tests', 'eql.etc'],
    package_data={
        'eql': etc_files,
    },
    zip_safe=False,
)
