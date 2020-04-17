"""Test case."""
import unittest
import itertools

from eql.ast import Field, Null, FunctionCall
from eql.errors import EqlTypeMismatchError
from eql.parser import parse_expression, implied_booleans
from eql.types import TypeFoldCheck, TypeHint, NodeInfo

from . import utils


class TestTypeSystem(unittest.TestCase):
    """Test that the type system correctly validates types."""

    def test_specifier_checks(self):
        """Test that specifiers are properly compared."""
        dynamic = TypeFoldCheck(TypeHint.Unknown, False)
        literal = TypeFoldCheck(TypeHint.Unknown, True)
        either = TypeHint.Unknown

        dynamic_node = NodeInfo(Field("a"), TypeHint.Numeric)
        self.assertTrue(dynamic_node.validate_literal(dynamic))
        self.assertFalse(dynamic_node.validate_literal(literal))
        self.assertTrue(dynamic_node.validate_literal(either))

        literal_node = NodeInfo(Null(), TypeHint.Numeric)
        self.assertFalse(literal_node.validate_literal(dynamic))
        self.assertTrue(literal_node.validate_literal(literal))
        self.assertTrue(literal_node.validate_literal(either))

        unfolded = NodeInfo(FunctionCall("length", [Null()]), TypeHint.Numeric)
        self.assertFalse(unfolded.validate_literal(dynamic))
        self.assertFalse(unfolded.validate_literal(literal))
        self.assertTrue(unfolded.validate_literal(either))

    def test_type_checks(self):
        """Test that types are properly compared."""
        tests = [
            (TypeHint.String, TypeHint.String, True),
            (TypeHint.String, TypeHint.Boolean, False),
            (TypeHint.Numeric, TypeHint.String, False),

            # anything could potentially be null
            (TypeHint.Null, TypeHint.Numeric, True),
            (TypeHint.String, TypeHint.Null, True),
            (TypeHint.Null, TypeHint.Null, True),

            # test out unions
            (TypeHint.String, (TypeHint.Numeric, TypeHint.Null), True),
            (TypeHint.String, TypeHint.primitives(), True),
        ]

        for hint1, hint2, expected in tests:
            output = NodeInfo(None, hint1).validate_type(hint2)
            self.assertEqual(output, expected, "hint {}.validate({}) != {}".format(hint1, hint2, expected))

    def test_type_match_comparisons(self):
        """Test that all valid non-null type comparisons."""
        comparables = [
            [utils.random_bool],
            [utils.random_int, utils.random_float],
            [utils.random_string]
        ]

        for row in comparables:
            for left_getter, right_getter in itertools.product(row, repeat=2):
                lv = left_getter()
                rv = right_getter()
                left = utils.unfold(lv)
                right = utils.unfold(rv)

                def parse_op(op):
                    parse_expression("{} {} {}".format(left, op, right))

                parse_op("==")
                parse_op("!=")

                for op in ("<", "<=", ">=", ">"):
                    if isinstance(lv, bool):
                        # booleans can't be compared with inequalities
                        with self.assertRaises(EqlTypeMismatchError):
                            parse_op(op)
                    else:
                        parse_op(op)

    def test_type_mismatch_comparisons(self):
        """Check that improperly compared types raise mismatch errors."""
        comparables = {
            (float, int),
            (int, float),
            (bool, bool),
            (str, str),
        }

        get_values = [utils.random_bool, utils.random_int, utils.random_float, utils.random_string]

        for lhs_getter, rhs_getter in itertools.product(get_values, repeat=2):
            lv = lhs_getter()
            rv = rhs_getter()

            # skip over types that we know will match
            if type(lv) == type(rv) or (type(lv), type(rv)) in comparables:
                continue

            left = utils.unfold(lv)
            right = utils.unfold(rv)

            for comparison in ["==", "!=", "<", "<=", ">=", ">"]:
                with self.assertRaises(EqlTypeMismatchError):
                    parse_expression("{left} {comp} {right}".format(left=left, comp=comparison, right=right))

    def test_parse_implied_booleans(self):
        """Test that parsing with implicit boolean casting works as expected."""
        with implied_booleans:
            for num_bools in range(2, 10):
                values = [utils.unfold(utils.random_value()) for _ in range(num_bools)]

                parse_expression(" and ".join(values))
                parse_expression(" or ".join(values))

    def test_invalid_function_signature(self):
        """Check that function signatures are correct."""
        expected_type_mismatch = [
            "length(0)",
            "wildcard(abc, def)",
            "length(f) > 'def'",
        ]

        for expression in expected_type_mismatch:
            self.assertRaises(EqlTypeMismatchError, parse_expression, expression)
