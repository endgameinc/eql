"""Tests for optimization of syntax trees."""
import unittest

from eql.ast import *  # noqa: F403
from eql.parser import parse_expression, skip_optimizations


class TestParseOptimizations(unittest.TestCase):
    """Tests that the parser returns optimized syntax trees."""

    def test_set_static_optimizations(self):
        """Check that checks for static fields in sets return optimized ASTs."""
        expression = '"something" in ("str", "str2", "str3", "str4", someField)'
        optimized = '"something" == someField'
        self.assertEqual(parse_expression(expression), parse_expression(optimized))

        expression = '"something" in ("str", "str2", "str3", "str4", field1, field2)'
        optimized = '"something" in (field1, field2)'
        self.assertEqual(parse_expression(expression), parse_expression(optimized))

    def test_set_optimizations(self):
        """Test that set unions, intersections, etc. are correct."""
        duplicate_values = parse_expression('fieldname in ("a", "b", "C", "d", 1, "d", "D", "c")')
        no_duplicates = parse_expression('fieldname in ("a", "b", "C", "d", 1)')
        self.assertEqual(duplicate_values, no_duplicates, "duplicate values were not removed")

        two_sets = parse_expression('fieldname in ("a", "b", "C", "x") and fieldname in ("d", "c", "g", "X")')
        intersection = parse_expression('fieldname in ("C", "x")')
        self.assertEqual(two_sets, intersection, "intersection test failed")

        two_sets = parse_expression('(fieldname in ("a", "b", "C", "x")) and fieldname in ("d", "f", "g", 123)')
        self.assertEqual(two_sets, Boolean(False), "empty intersection test failed")

        two_sets = parse_expression('fieldname in ("a", "b", "C", "x") or fieldname in ("d", "c", "g", "X")')
        union = parse_expression('fieldname in ("a", "b", "C", "x", "d", "g")')
        self.assertEqual(two_sets, union, "union test failed")

        literal_check = parse_expression('"ABC" in ("a", "ABC", "C")')
        self.assertEqual(literal_check, Boolean(True), "literal comparison failed")

        literal_check = parse_expression('"def" in ("a", "ABC", "C")')
        self.assertEqual(literal_check, Boolean(False), "literal comparison failed")

        dynamic_values = parse_expression('"abc" in ("a", "b", fieldA, "C", "d", fieldB, fieldC)')
        no_duplicates = parse_expression('"abc" in (fieldA, fieldB, fieldC)')
        self.assertEqual(dynamic_values, no_duplicates, "literal values were not removed")

        dynamic_values = parse_expression('fieldA in ("a", "b", "C", "d", fieldA, fieldB, fieldC)')
        self.assertEqual(dynamic_values, Boolean(True), "dynamic set lookup not optimized")

        and_not = parse_expression('NAME in ("a", "b", "c", "d") and not NAME in ("b", "d")')
        subtracted = parse_expression('NAME in ("a", "c")')
        self.assertEqual(and_not, subtracted, "set subtraction failed")

    def test_compound_merging_sets(self):
        """Test that compound boolean terms are merged correctly."""
        mixed_sets = parse_expression('opcode=1 and name in ("a", "b", "c", "d") and name in ("b", "d")')
        optimized = parse_expression('opcode=1 and name in ("b", "d")')
        self.assertEqual(mixed_sets, optimized, "failed to merge at tail of AND")

        mixed_sets = parse_expression('opcode=1 and name in ("a", "b", "c", "d") and name in ("b", "d") and x=1')
        optimized = parse_expression('opcode=1 and name in ("b", "d") and x=1')
        self.assertEqual(mixed_sets, optimized, "failed to merge at middle of AND")

        mixed_sets = parse_expression('opcode=1 or name in ("a", "b", "c", "d") or name in ("e", "f")')
        optimized = parse_expression('opcode=1 or name in ("a", "b", "c", "d", "e", "f")')
        self.assertEqual(mixed_sets, optimized, "failed to merge at tail of OR")

        mixed_sets = parse_expression('opcode=1 or name in ("a", "b", "c", "d") or name in ("e", "f") or x=1')
        optimized = parse_expression('opcode=1 or name in ("a", "b", "c", "d", "e", "f") or x=1')
        self.assertEqual(mixed_sets, optimized, "failed to merge at middle of OR")

    def test_comparisons_to_sets(self):
        """Test that multiple comparisons become sets."""
        multi_compare = parse_expression('pid == 4 or pid == 8 or pid == 520')
        optimized = parse_expression("pid in (4, 8, 520)")
        self.assertEqual(multi_compare, optimized, "Failed to merge comparisons into a set")

    def test_set_comparison_optimizations(self):
        """Test that sets and comparisons are merged."""
        set_or_comp = parse_expression('name in ("a", "b") or name == "c"')
        optimized = parse_expression('name in ("a", "b", "c")')
        self.assertEqual(set_or_comp, optimized, "Failed to OR a set with matching comparison")

        set_and_comp = parse_expression('name in ("a", "b") and name == "c"')
        optimized = parse_expression('false')
        self.assertEqual(set_and_comp, optimized, "Failed to AND a set with matching missing comparison")

        set_and_comp = parse_expression('name in ("a", "b") and name == "b"')
        optimized = parse_expression('name == "b"')
        self.assertEqual(set_and_comp, optimized, "Failed to AND a set with matching comparison")

        # switch the order
        comp_or_set = parse_expression('name == "c" or name in ("a", "b")')
        optimized = parse_expression('name in ("c", "a", "b")')
        self.assertEqual(comp_or_set, optimized, "Failed to OR a comparison with a matching set")

        comp_and_set = parse_expression('name == "c" and name in ("a", "b")')
        optimized = parse_expression('false')
        self.assertEqual(comp_and_set, optimized, "Failed to AND a comparison with a matching missing set")

        comp_and_set = parse_expression('name == "b" and name in ("a", "b")')
        optimized = parse_expression('name == "b"')
        self.assertEqual(comp_and_set, optimized, "Failed to AND a comparison with a matching set")

        # test that values can be subtracted individually from sets
        set_and_not = parse_expression('name in ("a", "b", "c") and name != "c"')
        optimized = parse_expression('name in ("a", "b")')
        self.assertEqual(set_and_not, optimized, "Failed to subtract specific value from set")

    def test_static_value_optimizations(self):
        """Test parser optimizations for comparing static values."""
        expected_true = [
            '10 == 10',
            '10 == 10.0',
            '"abc" == "abc"',
            'true == true',
            'true != false',
            '"" == ""',
            '"" == "*"',
            '"aaaaa" == "*"',
            '"abc" == "*abc*"',
            '"abc" == "*ABC*"',
            '"ABC" == "*abc*"',
            '"abc" != "d*"',
            '"net view" == "net* view*"',
            '"net view" == "net* view"',
            '"net view view" == "net* view"',
            '"net   view " == "net* VIEW*"',
            '"Net!!! VIEW    view net view" == "net* view*"',
            'not "Net!!! VIEW    view net view" != "net* view*"',
            '"Newww!!! VIEW    view net view" != "net* view*"',
            '1 < 2',
            '1 <= 2',
            '2 <= 2',
            '1 <= 1.0',
            '1.0 <= 1',
            '2 > 1',
            '2 >= 1',
            '2 >= 2',
            '2 != 1',
            '(1 * 2 + 3 * 4 + 10 / 2) == (2 + 12 + 5)',
            '(1 * 2 + 3 * 4 + 10 / 2) == 19',
            '1 * 2 + 3 * 4 + 10 / 2 == 2 + 12 + 5',
            '"ABC" <= "ABC"',
            "length('abcdefg') == 7",
            "100 in (1, 2, 3, 4, 100, 105)",
            "'rundll' in (abc.def[100], 'RUNDLL')",
            "not 'rundll' in ('100', 'nothing')",
            '1 - -2 == 3',
            '1 - +2 == -1',
            '1 +- length(a) == 1 - length(a)',
            '100:concat():length() == 3',
            '995 == (100 * 10):subtract("hello":length())',
            'cidrMatch("192.168.13.5", "192.168.0.0/16")',
        ]

        expected_false = [
            '"b" == "a"',
            '1 == 2',
            '1 > 2',
            '5 <= -3',
            '"ABC" = "abcd"',
            '"ABC*DEF" == " ABC    DEF    "',
            '"abc" > "def"',
            '"abc" != "abc"',
            'cidrMatch("1.2.3.4", "192.168.0.0/16")',
            # check that these aren't left to right
            '1 * 2 + 3 * 4 + 10 / 2 == 15',
        ]

        for expression in expected_true:
            ast = parse_expression(expression)
            self.assertIsInstance(ast, Boolean, 'Failed to optimize {}'.format(expression))
            self.assertTrue(ast.value, 'Parser did not evaluate {} as true'.format(expression))

        for expression in expected_false:
            ast = parse_expression(expression)
            self.assertIsInstance(ast, Boolean, 'Failed to optimize {}'.format(expression))
            self.assertFalse(ast.value, 'Parser did not evaluate {} as false'.format(expression))

    def test_unoptimized(self):
        """Test that optimization can be turned off."""
        with skip_optimizations:
            self.assertEqual(parse_expression("1 + 2"), MathOperation(Number(1), "+", Number(2)))
            self.assertEqual(parse_expression("1 + 2"), MathOperation(Number(1), "+", Number(2)))
