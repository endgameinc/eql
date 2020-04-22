"""Test Python Engine for EQL."""
import unittest

from eql.ast import Literal
from eql.engine import PythonEngine
from eql.parser import parse_expression, skip_optimizations

from . import utils


class TestNullPropagation(unittest.TestCase):
    """Test correctness of null handling through the optimizer and Python Engine."""

    @staticmethod
    def engine_eval(expr):
        """Evaluate an unoptimized expression in the ``PythonEngine``."""
        return PythonEngine().convert(expr)(None)

    def assert_fold(self, text, output):
        """Helper method for checking that expressions fold and evaluate correctly."""
        with skip_optimizations:
            parsed = parse_expression(text)

        self.assertNotIsInstance(parsed, Literal)
        self.assertEqual(self.engine_eval(parsed), output, "for: {} == {}".format(text, output))
        self.assertEqual(parsed.fold(), output, "for: {} == {}".format(text, output))

    def test_null_or_handling(self):
        """Test that nulls are correctly propagated through ``or`` for the engine and optimizer."""
        expected = [
            ("true  or false", True),
            ("true  or null ", True),
            ("true  or true ", True),
            ("false or null ", None),
            ("false or false", False),
            ("false or true ", True),
            ("null  or false", None),
            ("null  or null ", None),
            ("null  or true ", True),
        ]

        for text, output in expected:
            self.assert_fold(text, output)

    def test_null_and_handling(self):
        """Test that nulls are correctly propagated through ``and``."""
        expected = [
            ("true  and false",  False),
            ("true  and null ", None),
            ("true  and true ", True),
            ("false and null ", False),
            ("false and false", False),
            ("false and true ", False),
            ("null  and false", False),
            ("null  and null ", None),
            ("null  and true ", None),
        ]

        for text, output in expected:
            self.assert_fold(text, output)

    def test_null_compares(self):
        """Test that all types can be compared to null."""
        values = [None, utils.random_bool(), utils.random_int(), utils.random_string(), utils.random_float()]

        for v in values:
            unparsed = utils.unfold(v)

            def folds_to(fmt_string, expected):
                self.assert_fold(fmt_string.format(unparsed), expected)

            folds_to("{}   == null", v is None)
            folds_to("{}   != null", v is not None)
            folds_to("null ==   {}", v is None)
            folds_to("null !=   {}", v is not None)

            if not isinstance(v, bool):
                folds_to("{} <    null", None)
                folds_to("{}   <= null", None)
                folds_to("{}   >  null", None)
                folds_to("{}   >= null", None)
                folds_to("null <    {}", None)
                folds_to("null <=   {}", None)
                folds_to("null >=   {}", None)
                folds_to("null >    {}", None)

    def test_not_null(self):
        """Test that ``not null`` returns ``null``."""
        expected = [
            ("not true", False),
            ("not false", True),
            ("not null", None),
        ]

        for text, output in expected:
            self.assert_fold(text, output)
