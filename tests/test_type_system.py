"""Test case."""
import unittest

from eql.errors import EqlTypeMismatchError
from eql.parser import parse_expression
from eql.types import *  # noqa


class TestTypeSystem(unittest.TestCase):
    """Test that the type system correctly validates types."""

    def test_specifier_checks(self):
        """Test that specifiers are properly compared."""
        expected = [
            # Full truth table
            (DYNAMIC_SPECIFIER, NO_SPECIFIER, False),
            (DYNAMIC_SPECIFIER, LITERAL_SPECIFIER, False),
            (DYNAMIC_SPECIFIER, DYNAMIC_SPECIFIER, True),

            (LITERAL_SPECIFIER, NO_SPECIFIER, False),
            (LITERAL_SPECIFIER, LITERAL_SPECIFIER, True),
            (LITERAL_SPECIFIER, DYNAMIC_SPECIFIER, False),

            (NO_SPECIFIER, DYNAMIC_SPECIFIER, True),
            (NO_SPECIFIER, LITERAL_SPECIFIER, True),
        ]

        for spec1, spec2, rv in expected:
            self.assertEqual(check_specifiers(spec1, spec2), rv, "specifier {} x {} != {}".format(spec1, spec2, rv))

    def test_type_checks(self):
        """Test that types are properly compared."""
        tests = [
            (BASE_STRING, BASE_STRING, True),
            (BASE_STRING, BASE_BOOLEAN, False),
            (BASE_NUMBER, BASE_STRING, False),

            # anything could potentially be null
            (BASE_NULL, BASE_NUMBER, False),
            (BASE_STRING, BASE_NULL, False),
            (BASE_NULL, BASE_NULL, True),

            # test out unions
            (BASE_STRING, (BASE_NUMBER, BASE_NULL), False),
            ((BASE_STRING, (BASE_NUMBER, (BASE_STRING, BASE_BOOLEAN))), BASE_NULL, False),
            ((BASE_STRING, (BASE_NUMBER, (BASE_STRING, BASE_BOOLEAN))), BASE_BOOLEAN, True),
            (BASE_ALL, BASE_STRING, True),
            (BASE_STRING, BASE_STRING, True),
            (BASE_PRIMITIVES, BASE_STRING, True),
            ((BASE_NUMBER, BASE_STRING), BASE_BOOLEAN, False),
            ((BASE_NUMBER, (BASE_PRIMITIVES, ), BASE_BOOLEAN), BASE_BOOLEAN, True)
        ]

        for hint1, hint2, expected in tests:
            output = check_types(hint1, hint2)
            self.assertEqual(output, expected, "hint {} x {} != {}".format(hint1, hint2, expected))

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
            "(process_path or process_name) == '*net.exe'",
            "'hello' < 'hELLO'",
            "1 < 2",
            "(data and data.alert_details and data.alert_details.process_path) == 'net.exe'",
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
            # no longer invalid
            # "concat(1, 2, null)",

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
