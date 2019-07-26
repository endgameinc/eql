"""Test case."""
import unittest

from eql.ast import *  # noqa: F403
from eql.pipes import *  # noqa: F403
from eql.parser import (
    parse_expression
)
from eql.walkers import Walker, RecursiveWalker


class TestAbstractSyntaxTree(unittest.TestCase):
    """Test EQL parsing."""

    def test_abstract_methods(self):
        """Test that abstract methods are raising exceptions."""
        node = EqlNode()
        self.assertRaises(NotImplementedError, node.render)

        macro = BaseMacro("name")
        self.assertRaises(NotImplementedError, macro.expand, [])

    def test_invalid_ast(self):
        """Test that invalid ast nodes raise errors."""
        self.assertRaises(TypeError, Literal, True)
        self.assertRaises(TypeError, Literal, dict())
        self.assertRaises(TypeError, Literal, list())
        self.assertRaises(TypeError, Literal, complex())
        self.assertRaises(TypeError, Literal, object())
        self.assertRaises(TypeError, Literal, lambda: None)
        self.assertRaises(TypeError, Literal, object)

    def test_literals(self):
        """Test that literals are parsed correctly."""
        eql_literals = [
            ('true', True, Boolean),
            ('false', False, Boolean),
            ('100', 100, Number),
            ('1.5', 1.5, Number),
            ('.6', .6, Number),
            ('-100', -100, Number),
            ('-15.24', -15.24, Number),
            ('"100"', "100", String),
            ('null', None, Null),
        ]
        for text, expected_value, expected_type in eql_literals:
            node = parse_expression(text)
            rendered = node.render()
            re_parsed = parse_expression(rendered)
            self.assertIsInstance(node, expected_type)
            self.assertEqual(node.value, expected_value)
            self.assertEqual(node, re_parsed)

    def test_camelized(self):
        """Test camelization of class names."""
        camelized = Walker.camelized
        self.assertEqual(camelized(String), "string")
        self.assertEqual(camelized(EventQuery), "event_query")
        self.assertEqual(camelized(EventQuery), "event_query")
        self.assertEqual(camelized(EventQuery), "event_query")
        self.assertEqual(camelized(FunctionCall), "function_call")
        self.assertEqual(camelized(PipeCommand), "pipe_command")
        self.assertEqual(camelized(UniqueCountPipe), "unique_count_pipe")

    def test_walker(self):
        """Check that walker transformation works properly."""
        walker = RecursiveWalker()
        node = parse_expression("process_name == 'net.exe' or file_name == 'abc.txt'")

        def assert_deep_copy(a, b):
            """Check that deep copies are created."""
            self.assertEqual(a, b)
            self.assertIsNot(a, b)

            for deep_a, deep_b in zip(a, b):
                self.assertEqual(deep_a, deep_b)
                self.assertIsNot(deep_a, deep_b)

        assert_deep_copy(node, walker.copy_node(node))

        class SimpleWalker(RecursiveWalker):

            def _walk_comparison(self, node):
                if node.left == Field('file_name'):
                    return self.walk(parse_expression('user_name == "TEMP_USER"'))
                return self._walk_base_node(node)

            def _walk_string(self, node):
                if node == String("TEMP_USER"):
                    return String("artemis")
                return node

        walker = SimpleWalker()
        expected = parse_expression('process_name == "net.exe" or user_name == "artemis"')
        self.assertEqual(walker.walk(node), expected)
