#! /usr/bin/env python

"""Perform setup of the package for build."""
import sys
import glob
import os
import re
import io

from setuptools import setup, Command
from setuptools.command.test import test as TestCommand


with io.open('eql/__init__.py', 'rt', encoding='utf8') as f:
    __version__ = re.search(r'__version__ = \'(.*?)\'', f.read()).group(1)

install_requires = [
    "lark-parser~=0.12.0",
    "enum34; python_version<'3.4'",
]

test_requires = [
    "mock~=1.3.0",
    "pytest~=3.8.2",
    "pytest-cov==2.4",
    "flake8==2.5.1",
    "pep257==0.7.0",
    "coverage==4.5.3",
    "flake8-pep257==1.0.5",
    "PyYAML<6.0; python_version<'3.4'",
    "PyYAML; python_version>='3.4'",
    "toml~=0.10",
    "pluggy==1.0.0-dev0; python_version<'3.4'",
    "configparser<5.0; python_version<'3.4'",
    "contextlib2~=0.6.0",
    "more-itertools~=5.0; python_version<'3.4'",
    "importlib-metadata<3.0; python_version<'3.4'",
    "zipp<1.0; python_version<'3.4'",
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
        self.distribution.fetch_build_eggs(test_requires)
        self.distribution.packages.append('tests')

        from flake8.main import Flake8Command
        flake8cmd = Flake8Command(self.distribution)
        flake8cmd.options_dict = {}
        flake8cmd.run()


class Test(TestCommand):
    """Use pytest (http://pytest.org/latest/) in place of the standard unittest library."""

    user_options = [("pytest-args=", "a", "Arguments to pass to pytest")]

    def initialize_options(self):
        """Need to ensure pytest_args exists."""
        TestCommand.initialize_options(self)
        self.pytest_args = []

    def run_tests(self):
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
        'Programming Language :: Python :: 2',
        'Programming Language :: Python :: 2.7',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.4',
        'Programming Language :: Python :: 3.5',
        'Programming Language :: Python :: 3.6',
        'Programming Language :: Python :: 3.7',
        'Topic :: Database',
        'Topic :: Internet :: Log Analysis',
        'Topic :: Scientific/Engineering :: Information Analysis',
    ],
    url='https://eql.readthedocs.io',
    tests_require=install_requires + test_requires,
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
