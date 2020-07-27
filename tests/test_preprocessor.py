"""Tests for the EQL preprocessor."""
import os
import unittest
from collections import OrderedDict

from eql.ast import *  # noqa: F403
from eql.parser import *  # noqa: F403
from eql.transpilers import TextEngine
from eql.errors import EqlTypeMismatchError
from eql.schema import Schema


class TestPreProcessor(unittest.TestCase):
    """Tests for the EQL PreProcessor."""

    macro_definitions = """
        macro A_OR_B(a,b)
            a or b

        macro XOR(a,b)
            A_OR_B(a and not b, b and not a)

        macro IN_GRAYLIST(proc)
            proc in (
                "msbuild.exe",
                "powershell.exe",
                "cmd.exe",
                "netsh.exe"
            )

        macro PROCESS_IN_GRAYLIST()
            IN_GRAYLIST(process_name)

        macro PARENT_XOR_CHILD_IN_GRAYLIST()
            XOR(IN_GRAYLIST(process_name), IN_GRAYLIST(parent_process_name))

        macro DESCENDANT_OF_PROC(expr)
            descendant of [process where opcode==1 and expr]
    """

    def test_parse_with_preprocessor(self):
        """Test that preprocessor works with the parser."""
        preprocessor = get_preprocessor("""
        const ABC = 123
        const DEF = 456
        const GHI = 123

        macro COMPARE_TWO(a, b)  a == b
        macro GET_TRUE(a)   COMPARE_TWO(a, a)
        macro IS_123(a)     a == ABC
        """)

        def p(text):
            return parse_expression(text, preprocessor=preprocessor)

        self.assertEqual(p('ABC'), Number(123))
        self.assertEqual(p('COMPARE_TWO(some_field, "abc.exe")'), p('some_field == "abc.exe"'))
        self.assertEqual(p('COMPARE_TWO(105, 105)'), Boolean(True))
        self.assertEqual(p('GET_TRUE(100)'), Boolean(True))
        self.assertEqual(p('IS_123(456)'), Boolean(False))
        self.assertEqual(p('IS_123(123)'), Boolean(True))

        # now double up
        double_pp = get_preprocessor("""
        macro TRUE()    GET_TRUE(105)
        macro FALSE()   not TRUE()
        """, preprocessor=preprocessor)

        def pp(text):
            return parse_expression(text, preprocessor=double_pp)

        self.assertEqual(pp('ABC'), Number(123))
        self.assertEqual(pp('TRUE()'), Boolean(True))
        self.assertEqual(pp('FALSE()'), Boolean(False))
        self.assertEqual(pp('not FALSE()'), Boolean(True))

    def test_preprocessor_type_hints(self):
        """Test that type hints are correct for when parsing with a preprocessor."""
        preprocessor = get_preprocessor("""
        macro ENUM_COMMAND(name)
          name in ("net.exe", "whoami.exe", "hostname.exe")
        macro CONSTANT()        1
        """)

        with preprocessor:
            parse_query("process where ENUM_COMMAND(process_name)")
            parse_query("process where true | filter ENUM_COMMAND(process_name)")
            parse_query("process where true | unique ENUM_COMMAND(process_name)")
            parse_query("process where true | filter CONSTANT()")

            # unique requires a dynamic type, but there are no fields in CONSTANT
            with self.assertRaisesRegex(EqlTypeMismatchError, "Expected dynamic number not literal number to unique"):
                parse_query("process where true | unique CONSTANT()")

    def test_macro_expansion(self):
        """Test EQL custom macros."""
        expanded = {
            "A_OR_B": "a or b",
            "XOR": "(a and not b) or (b and not a)",
            "IN_GRAYLIST": 'proc in ("msbuild.exe", "powershell.exe", "cmd.exe", "netsh.exe")',
            "PROCESS_IN_GRAYLIST": 'process_name in ("msbuild.exe", "powershell.exe", "cmd.exe", "netsh.exe")',
            "PARENT_XOR_CHILD_IN_GRAYLIST": (
                '(  '
                '   process_name in ("msbuild.exe", "powershell.exe", "cmd.exe", "netsh.exe") and not '
                '   parent_process_name in ("msbuild.exe", "powershell.exe", "cmd.exe", "netsh.exe")'
                ') or ('
                '   parent_process_name in ("msbuild.exe", "powershell.exe", "cmd.exe", "netsh.exe") and not '
                '   process_name in ("msbuild.exe", "powershell.exe", "cmd.exe", "netsh.exe")'
                ')'
            ),
            'DESCENDANT_OF_PROC': 'descendant of [process where opcode == 1 and expr]'
        }

        macros = parse_definitions(self.macro_definitions)
        lookup = OrderedDict()

        with ignore_missing_functions:
            for macro in macros:
                lookup[macro.name] = macro
                rendered = macro.render()
                macro_copy = parse_definition(rendered)
                self.assertEqual(macro, macro_copy)
                self.assertEqual(rendered, macro_copy.render(), "Macro doesn't render valid EQL.")

            # Now load up each macro to the engine
            engine = PreProcessor(macros)

            # Confirm that nested macros are expanded appropriately
            for name, macro in engine.macros.items():
                expected_expr = parse_expression(expanded[name])
                self.assertEqual(macro.expression, expected_expr)
                self.assertEqual(macro.expression.render(), expected_expr.render())

            # Expand some EQL queries
            queries = [
                ('process where DESCENDANT_OF_PROC(process_name="explorer.exe")',
                 'process where descendant of [process where opcode=1 and process_name == "explorer.exe"]'
                 ),
                ('process where XOR(a=="b", c=="d")',
                 'process where ((a == "b") and not (c == "d")) or  ((c == "d") and not (a == "b"))'
                 ),
                ('file where true',
                 'file where true',
                 ),
                ('process where opcode=1 and PROCESS_IN_GRAYLIST()',
                 'process where opcode==1 and process_name in ("msbuild.exe","powershell.exe","cmd.exe","netsh.exe")'
                 ),
            ]

            for query, expanded_query in queries:
                before_node = parse_query(query)
                actual = engine.expand(before_node)
                expected = parse_query(expanded_query)

                # Test that eval + repr works
                actual_repr = repr(actual)
                eval_actual = eval(actual_repr)

                self.assertEqual(actual, expected)
                self.assertEqual(eval_actual, actual)
                self.assertTrue(actual == expected)
                self.assertFalse(actual != expected)
                error_msg = "'{}' expanded to '{}' instead of '{}'".format(query, actual.render(), expected.render())
                self.assertEqual(actual.render(), expected.render(), error_msg)

            query = parse_expression("DESCENDANT_OF_PROC()")
            self.assertRaisesRegex(ValueError, r"Macro .+ expected \d+ arguments .*", engine.expand, query)

            query = parse_expression("DESCENDANT_OF_PROC(1,2,3)")
            self.assertRaisesRegex(ValueError, r"Macro .+ expected \d+ arguments .*", engine.expand, query)

    def test_custom_macro(self):
        """Test python custom macro expansion."""
        def optimize_length(args):
            arg, = args  # only 1 allowed
            if isinstance(arg, String):
                return Number(len(arg.value))
            else:
                return FunctionCall('length', [arg])

        macro = CustomMacro('LENGTH', optimize_length)
        engine = PreProcessor([macro])

        with ignore_missing_functions:
            example = parse_query('process where LENGTH("python.exe") == LENGTH(process_name)')
            expected = parse_query('process where 10 == length(process_name)')

        output = engine.expand(example)
        self.assertEqual(output, expected, "Custom macro LENGTH was not properly expanded")

        with ignore_missing_functions:
            example = parse_query('process where LENGTH("abc", "def")')

        self.assertRaisesRegex(ValueError, "too many values to unpack", engine.expand, example)

    def test_load_definitions_from_file(self):
        """Test loading definitions from a file."""
        filename = 'example-definitions.eql.tmp'
        config = {'definitions_files': [filename]}
        with open(filename, 'w') as f:
            f.write(self.macro_definitions)
        engine = TextEngine(config)
        os.remove(filename)
        self.assertGreater(len(engine.preprocessor.macros), 0, "Definitions failed to load")

    def test_mixed_definitions(self):
        """Test that macro and constant definitions can be loaded correctly."""
        defn = parse_definitions("""
        const magic = 100
        macro OR(a, b) a or b
        """)
        pp = PreProcessor(defn)

        # Confirm that copy and adding is working
        pp2 = pp.copy()
        pp.add_definition(parse_definition("macro ABC(a, b, c) error_error_error"))
        pp2.add_definition(parse_definition("macro ABC(a, b, c) concat(a, magic, c)"))

        matches = [
            ("abc", "abc"),
            ("OR(x, y)", "x or y"),
            ("magic", "100"),
            ("ABC(0,1,2)", "concat(0, 100, 2)"),
        ]

        for before, after in matches:
            with ignore_missing_functions:
                before = parse_expression(before)
            after = parse_expression(after)
            self.assertEqual(pp2.expand(before), after)

    def test_macro_expansion_hinting_bug(self):
        """Test bugfix for macro expansion."""
        preprocessor = get_preprocessor("macro SELF(a)   a")

        with Schema({"foo": {"bar": "number"}}), preprocessor:
            parse_query("foo where SELF(bar) == 1")

            with self.assertRaises(EqlTypeMismatchError):
                parse_query("foo where SELF(bar) == 'baz'")
