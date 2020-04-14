"""Test case."""
import unittest

from eql.errors import EqlTypeMismatchError
from eql.parser import parse_expression
from eql.types import TypeFoldCheck, TypeHint, NodeInfo
from eql.ast import Field, Null, FunctionCall


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
            (TypeHint.Null, TypeHint.Numeric, False),
            (TypeHint.String, TypeHint.Null, False),
            (TypeHint.Null, TypeHint.Null, True),

            # test out unions
            (TypeHint.String, (TypeHint.Numeric, TypeHint.Null), False),
            (TypeHint.String, TypeHint.primitives(), True),
        ]

        for hint1, hint2, expected in tests:
            output = NodeInfo(None, hint1).validate_type(hint2)
            self.assertEqual(output, expected, "hint {}.validate({}) != {}".format(hint1, hint2, expected))

    def test_parse_type_matches(self):
        """Check that improperly compared types are raising errors."""
        expected_type_match = [
            '1 or 2',
            'abc == null or def == null',
            "false or 1",
            "1 or 'abcdefg'",
            "false or 'string-false'",
            "port == 80 or command_line == 'defghi'",
            "(port != null or command_line != null)",
            "(process_path or process_name) == true",
            "'hello' < 'hELLO'",
            "1 < 2",
            "(data and data.alert_details and data.alert_details.process_path) == false",
        ]

        for expression in expected_type_match:
            parse_expression(expression)

    def test_parse_type_mismatches(self):
        """Check that improperly compared types are raising errors."""
        expected_type_mismatch = [
            '1 == "*"',
            'false = 1',
            '100 = "a"',
            '100 != "*abcdef*"',
            '100 in ("string1", "string2")',
            'true != 100',
            '100 != "abc"',
            '"some string" == null',
            'true < false',
            'true > "abc"',
            'field < true',
            'true <= 6',
            "'hello' > 500",

            "(process_path or process_name) == 'net.exe'",
            "(process_path and process_name) == 'net.exe'",

            # check for return types
            'true == length(abc)',
            '"true" == length(abc)',

            # check for mixed sets
            "'rundll' in (1, 2, 3, abc.def[100], 'RUNDLL', false)",
            "not 'rundll' in (1, 2, 3, '100', 'nothing', false)",
        ]

        for expression in expected_type_mismatch:
            self.assertRaises(EqlTypeMismatchError, parse_expression, expression)

    def test_invalid_function_signature(self):
        """Check that function signatures are correct."""
        expected_type_mismatch = [
            "length(0)",
            "wildcard(abc, def)",
            "length(f) > 'def'",
        ]

        for expression in expected_type_mismatch:
            self.assertRaises(EqlTypeMismatchError, parse_expression, expression)
