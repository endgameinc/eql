#! /usr/bin/env python

"""Perform setup of the package for build."""
import sys
import glob
import os
import re
import io

try:  # for pip >= 10
    from pip._internal.req import parse_requirements
except ImportError:  # for pip <= 9.0.3
    from pip.req import parse_requirements

from setuptools import setup, Command, find_packages
from setuptools.command.test import test as TestCommand


with io.open('eql/__init__.py', 'rt', encoding='utf8') as f:
    __version__ = re.search(r'__version__ = \'(.*?)\'', f.read()).group(1)


install_requires = parse_requirements('requirements.txt', session=False)
install_requires = [str(req.req) for req in install_requires]

test_requires = parse_requirements('requirements_test.txt', session=False)
test_requires = [str(req.req) for req in test_requires]


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

    def initialize_options(self):
        """Need to ensure pytest_args exists."""
        TestCommand.initialize_options(self)
        self.pytest_args = []

    def finalize_options(self):
        """Zero test_args and force test_suite to run."""
        TestCommand.finalize_options(self)
        self.test_args = [
            '--cov-report=xml', '--cov-report=html', '--cov=eql', '--junitxml=junit.xml', '-x', '-v'
        ]
        self.test_suite = True

    def run_tests(self):
        """Run pytest."""
        import pytest
        sys.exit(pytest.main(self.test_args))


etc_files = [os.path.relpath(fn, 'eql') for fn in glob.glob('eql/etc/*') if not fn.endswith('.py')]

setup(
    name='eql',
    version=__version__,
    description='Event Query Language',
    install_requires=install_requires,
    tests_require=test_requires,
    cmdclass={
        'lint': Lint,
        'test': Test
    },
    entry_points={
        'console_scripts': [
            'eql=eql.main:main',
        ],
    },
    extras_require={
        'lint': test_requires,
        'test': test_requires,
        'loaders': [
            'pyyaml',
            'toml',
        ]
    },
    packages=find_packages(),
    package_data={
        'eql': etc_files,
    },
    zip_safe=False,
)
