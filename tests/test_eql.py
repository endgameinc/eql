"""Test case."""
import datetime
import os
import sys
import traceback
import unittest
from collections import OrderedDict

from eql.ast import *  # noqa
from eql.engines.base import BaseEngine, TextEngine
from eql.errors import ParseError, SchemaError
from eql.parser import (
    parse_query, parse_expression, parse_definition, parse_definitions, parse_analytic, get_preprocessor
)
from eql.schema import use_schema


class TestEql(unittest.TestCase):
    """Test EQL parsing."""

    def test_abstract_methods(self):
        """Test that abstract methods are raising exceptions."""
        node = EqlNode()
        self.assertRaises(NotImplementedError, node.render)

        macro = BaseMacro("name")
        self.assertRaises(NotImplementedError, macro.expand, [])

    def test_invalid_ast(self):
        """Test that invalid ast nodes raise errors."""
        self.assertRaises(AssertionError, Literal, True)
        self.assertRaises(AssertionError, Literal, dict())
        self.assertRaises(AssertionError, Literal, list())
        self.assertRaises(AssertionError, Literal, complex())
        self.assertRaises(AssertionError, Literal, object())
        self.assertRaises(AssertionError, Literal, lambda: None)
        self.assertRaises(AssertionError, Literal, object)

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

    def test_valid_expressions(self):
        """Test that expressions are parsed correctly."""
        valid = [
            "1 == 1",
            "1 == (1 == 1)",
            'abc != "ghi"',
            "abc > 20",
            "f()",
            "somef(a,b,c,d,)",
            "a in (1,2,3,4,)",
            "f(abc) < g(hij)",
            "f(f(f(f(abc))))",
            'abc == f()',
            'f() and g()',
            "1",
            '(1)',
            "true",
            "false",
            "null",
            "not null",
            "abc",
            '"string"',
            'abc and def',
            '(1==abc) and def',
            'abc == (1 and 2)',
            'abc == (def and 2)',
            'abc == (def and def)',
            'abc == (def and ghi)',
            '"\\b\\t\\r\\n\\f\\\\\\"\\\'"',
        ]

        for query in valid:
            parse_expression(query)

    def test_functions(self):
        """Test that functions are being parsed correctly."""
        # Make sure that functions are parsing all arguments
        fn = parse_expression('somefunction('
                              '     a and c,'
                              '     false,'
                              '     d or g'
                              ')')
        self.assertIsInstance(fn, FunctionCall)
        self.assertEqual(len(fn.arguments), 3)

    def test_invalid_expressions(self):
        """Test that expressions are parsed correctly."""
        invalid = [
            'a xor b',  # made up comparator
            'def[ghi]',  # index not a number
            'def[-1]',  # negative indexes not supported
            'someFunc().abc',  # can't index these
            '1.2.3',  # invalid number
            'a.1',
            '()',  # nothing inside
            '',
            '"invalid"string"',
            '--100',
            '1000   100',
            '""    100',
            # literal values as fields
            'true.100',
            'null.abc',
            'abc[0].null',
            # require escape slashes,
            '\\R',
            '\\W',
        ]

        keywords = [
            'and', 'by', 'in', 'join', 'macro', 'not', 'of', 'or', 'sequence', 'until', 'where', 'with'
        ]

        for query in invalid:
            self.assertRaises(ParseError, parse_expression, query)

        for keyword in keywords:
            self.assertRaises(ParseError, parse_expression, keyword)
            parse_expression(keyword.upper())

    def test_valid_queries(self):
        """Make sure that EQL queries are properly parsed."""
        valid = [
            'file where true',
            'file where true and true',
            'file where false or true',
            'registry where not pid',
            'process where process_name == "net.exe" and command_line == "* user*.exe"',
            'process where command_line == "~!@#$%^&*();\'[]{}\\\\|<>?,./:\\"-= \' "',
            'process where \n\n\npid ==\t 4',
            'process where process_name in ("net.exe", "cmd.exe", "at.exe")',
            'process where command_line == "*.exe *admin*" or command_line == "* a b*"',
            'process where pid in (1,2,3,4,5,6,7,8) and abc == 100 and def == 200 and ghi == 300 and jkl == x',
            'process where ppid != pid',
            'image_load where not x != y',
            'image_load where not x == y',
            'image_load where not not not not x < y',
            'image_load where not x <= y',
            'image_load where not x >= y',
            'image_load where not x > y',
            'process where pid == 4 or pid == 5 or pid == 6 or pid == 7 or pid == 8',
            'network where pid == 0 or pid == 4 or (ppid == 0 or ppid = 4) or (abc == defgh) and process_name == "*" ',
            'network where pid = 4',
            'process where descendant of [process where process_name == "lsass.exe"] and process_name == "cmd.exe"',
            'join \t\t\t[process where process_name == "*"] [  file where file_path == "*"\n]',
            'join by pid [process where name == "*"] [file where path == "*"] until [process where opcode == 2]',
            'sequence [process where name == "*"] [file where path == "*"] until [process where opcode == 2]',
            'sequence by pid [process where name == "*"] [file where path == "*"] until [process where opcode == 2]',
            'join [process where process_name == "*"] by process_path [file where file_path == "*"] by image_path',
            'sequence [process where process_name == "*"] by process_path [file where file_path == "*"] by image_path',
            'sequence by pid [process where process_name == "*"] [file where file_path == "*"]',
            'sequence by pid with maxspan=200 [process where process_name == "*" ] [file where file_path == "*"]',
            'sequence by pid with maxspan=2s [process where process_name == "*" ] [file where file_path == "*"]',
            'sequence by pid with maxspan=2sec [process where process_name == "*" ] [file where file_path == "*"]',
            'sequence by pid with maxspan=2seconds [process where process_name == "*" ] [file where file_path == "*"]',
            'sequence with maxspan=2.5m [process where x == x] by pid [file where file_path == "*"] by ppid',
            'sequence by pid with maxspan=2.0h [process where process_name == "*"] [file where file_path == "*"]',
            'sequence by pid with maxspan=2.0h [process where process_name == "*"] [file where file_path == "*"]',
            'sequence by pid with maxspan=1.0075d [process where process_name == "*"] [file where file_path == "*"]',
            'dns where pid == 100 | head 100 | tail 50 | unique pid',
            'network where pid == 100 | unique command_line | count',
            'security where user_domain == "endgame" | count user_name a b | tail 5',
            'process where 1==1 | count user_name, unique_pid, myFn(field2,a,bc)',
            'process where 1==1 | unique user_name, myFn(field2,a,bc), field2',
            'registry where a.b',
            'registry where a[0]',
            'registry where a.b.c.d.e',
            'registry where a.b.c[0]',
            'registry where a[0].b',
            'registry where a[0][1].b',
            'registry where a[0].b[1]',
            'registry where topField.subField[100].subsubField == 0',
            'process where true | filter true',
            'process where 1==1 | filter abc == def',
            'process where 1==1 | filter abc == def and 1 != 2',
            'process where 1==1 | count process_name | filter percent > 0.5',
            'process where a > 100000000000000000000000000000000',
            'any where true | unique a b c | sort a b c | count',
            'any where true | unique a, b,   c | sort a b c | count',
            'any where true | unique a, b,   c | sort a,b,c | count',
            'file where child of [registry where true]',
            'file where event of [registry where true]',
            'file where event of [registry where true]',
            'file where descendant of [registry where true]',
            # multiple by values
            'sequence by field1  [file where true] by f1 [process where true] by f1',
            'sequence by a,b,c,d [file where true] by f1,f2 [process where true] by f1,f2',
            'sequence [file where 1] by f1,f2 [process where 1] by f1,f2 until [process where 1] by f1,f2',
            'sequence by f [file where true] by a,b [process where true] by c,d until [process where 1] by e,f',
            # sequence with named params
            'sequence by unique_pid [process where true] [file where true] fork',
            'sequence by unique_pid [process where true] [file where true] fork=true',
            'sequence by unique_pid [process where true] [file where true] fork=1',
            'sequence by unique_pid [process where true] [file where true] fork=false',
            'sequence by unique_pid [process where true] [file where true] fork=0 [network where true]',
            'sequence by unique_pid [process where true] [file where true] fork=0',
        ]

        datetime.datetime.now()

        for i, text in enumerate(valid):
            try:
                query = parse_query(text)
                rendered = query.render()
                self.assertEqual(text.split()[0], rendered.split()[0])

                # parse it again to make sure it's still valid and doesn't change
                parse_again = parse_query(rendered)
                rendered_again = parse_again.render()

                # repr + eval should also restore it properly
                # Test that eval + repr works
                actual_repr = repr(query)
                eval_actual = eval(actual_repr)

                self.assertEqual(query, parse_again, "Query didn't reparse correctly.")
                self.assertEqual(rendered, rendered_again)
                self.assertEqual(query, eval_actual)

            except ParseError:
                ex_type, ex, tb = sys.exc_info()
                traceback.print_exc(ex)
                traceback.print_tb(tb)
                self.fail("Unable to parse query #{}: {}".format(i, text))

    def test_invalid_schema(self):
        """Test that schema errors are being raised separately."""
        invalid = [
            'fakeNews where president == "russia"',
            'PROCESS where process_name == "bad.exe"',
            'Process where process_name == "bad.exe"',
            'file_ where process_name == "bad.exe"',
        ]
        for query in invalid:
            self.assertRaises(SchemaError, parse_query, query)

    def test_invalid_queries(self):
        """Test that invalid queries throw the proper error."""
        invalid = [
            'process where process_name == "abc.exe"     garbage extraneous \"input\"',
            'garbage process where process_name < "abc.e"xe"',
            'process',
            'process where abc == "extra"quote"',
            'file where and',
            'file where file_name and',
            'file_name and',
            'file_name )',
            'file_name (\r\n\r\n',
            'file_name where (\r\n\r\n)',
            'process where _badSymbol == 100',
            'process where 1field == 2field',
            'sequence where 1field == 2field',
            'process where true | filter',
            'process where true | badPipe',
            'process where true | badPipe a b c',
            'process where true | head -100',
            'process where descendant of []',
            'file where nothing of [process where true]',
            'file where DescenDant of [process where true]',
            'garbage',
            'process where process_name == "abc.exe" | count 100',
            'process where process_name == "abc.exe" | unique 100',
            'process where process_name == "abc.exe" | sort 100',
            'process where process_name == "abc.exe" | head 100 abc',
            'process where process_name == "abc.exe" | head abc',
            'process where process_name == "abc.exe" | head abc()',
            'process where process_name == "abc.exe" | head abc(def, ghi)',
            'sequence [process where pid == pid]',
            'sequence [process where pid == pid] []',
            'sequence with maxspan=false [process where true] [process where true]',
            'sequence with badparam=100 [process where true] [process where true]',
            # check that the same number of BYs are in every subquery
            'sequence [file where true] [process where true] by field1',
            'sequence [file where true] by field [file where true] by field1 until [file where true]',
            'sequence by a,b,c [file where true] by field [file where true] by field1 until [file where true]',
            'sequence [file where 1] by field [file where 1] by f1 until [file where 1] by f1,f2 | unique field',
            'sequence [process where 1] fork=true [network where 1]',
            'sequence [process where 1] [network where 1] badparam=true',
            'sequence [process where 1] [network where 1] fork=true fork=true',
            'sequence [process where 1] [network where 1] fork fork',
            'process where descendant of [file where true] bad=param',
            '| filter true'
        ]
        for query in invalid:
            self.assertRaises(ParseError, parse_query, query)

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
            if name == 'safePath':
                continue
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
        self.assertRaisesRegexp(ValueError, "Macro .+ expected \d+ arguments .*", engine.expand, query)

        query = parse_expression("DESCENDANT_OF_PROC(1,2,3)")
        self.assertRaisesRegexp(ValueError, "Macro .+ expected \d+ arguments .*", engine.expand, query)

    def test_engine_schema(self):
        """Test loading the engine with a custom schema."""
        queries = [
            'movie where name == "*Breakfast*" and IN_80s(release)',
            'person where name == "John Hughes"',
        ]

        analytic_dicts = [{'query': q} for q in queries]
        definitions = """
        macro IN_80s(date) date == "*/*/1980"
        """

        config = {
            'schema': {'event_types': {'movie': 1, 'person': 2}},
            'definitions': parse_definitions(definitions),
            'analytics': analytic_dicts
        }

        pp = PreProcessor()
        pp.add_definitions(config['definitions'])

        with use_schema(config['schema']):
            expected = [parse_analytic(d, preprocessor=pp) for d in analytic_dicts]

        engine = BaseEngine(config)
        with use_schema(engine.schema):
            engine.add_analytics([parse_analytic(d) for d in analytic_dicts])

        self.assertListEqual(engine.analytics, expected, "Analytics were not loaded and expanded properly.")

    def test_custom_macro(self):
        """Test python custom macro expansion."""
        def optimize_length(args, walker):
            arg, = args  # only 1 allowed
            if isinstance(arg, String):
                return Number(len(arg.value))
            else:
                return FunctionCall('length', [arg])

        macro = CustomMacro('LENGTH', optimize_length)
        engine = PreProcessor([macro])

        example = parse_query('process where LENGTH("python.exe") == LENGTH(process_name)')
        expected = parse_query('process where 10 == length(process_name)')

        output = engine.expand(example)
        self.assertEqual(output, expected, "Custom macro LENGTH was not properly expanded")

        example = parse_query('process where LENGTH("abc", "def")')
        self.assertRaisesRegexp(ValueError, "too many values to unpack", engine.expand, example)

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
        pp2.add_definition(parse_definition("macro ABC(a, b, c) f(a, magic, c)"))

        matches = [
            ("abc", "abc"),
            ("OR(x, y)", "x or y"),
            ("magic", "100"),
            ("ABC(0,1,2)", "f(0, 100, 2)"),
        ]
        for before, after in matches:
            before = parse_expression(before)
            after = parse_expression(after)
            self.assertEqual(pp2.expand(before), after)

    def test_static_value_optimizations(self):
        """Test parser optimizations for comparing static values."""
        expected_true = [
            '10 == 10',
            '10 == 10.0',
            '"abc" == "abc"',
            'true == true',
            'true != false',
            'true != 100',
            '100 != "abc"',
            '"" == ""',
            '"" == "*"',
            '"aaaaa" == "*"',
            '100 != "*abcdef*"',
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
            '"ABC" <= "ABC"',
            "length('abcdefg') == 7",
            "100 in (1, 2, 3, 4, 100, 105)",
            "'rundll' in (1, 2, 3, abc.def[100], 'RUNDLL', false)",
            "not 'rundll' in (1, 2, 3, '100', 'nothing', false)",
        ]

        for expression in expected_true:
            ast = parse_expression(expression)
            self.assertIsInstance(ast, Boolean, 'Failed to optimize {}'.format(expression))
            self.assertTrue(ast.value, 'Parser did not evaluate {} as true'.format(expression))

        expected_false = [
            '100 = "a"',
            '"b" == "a"',
            '1 == 2',
            '1 > 2',
            '5 <= -3',
            '"ABC" = "abcd"',
            '"ABC*DEF" == " ABC    DEF    "',
            '1 == "*"',
            '"abc" > "def"',
            '"abc" != "abc"',
        ]

        for expression in expected_false:
            ast = parse_expression(expression)
            self.assertIsInstance(ast, Boolean, 'Failed to optimize {}'.format(expression))
            self.assertFalse(ast.value, 'Parser did not evaluate {} as false'.format(expression))

        expression = '"something" in ("str", "str2", "str3", "str4", someField)'
        optimized = '"something" == someField'
        self.assertEqual(parse_expression(expression), parse_expression(optimized))

        expression = '"something" in ("str", "str2", "str3", "str4", field1, field2)'
        optimized = '"something" in (field1, field2)'
        self.assertEqual(parse_expression(expression), parse_expression(optimized))

    def test_query_events(self):
        """Test that event queries work with events[n].* syntax in pipes."""
        base_queries = ['abc', 'abc[123]', 'abc.def.ghi', 'abc.def[123].ghi[456]']
        for text in base_queries:
            field_query = parse_expression(text)  # type: Field
            events_query = parse_expression('events[0].' + text)  # type: Field

            index, query = field_query.query_multiple_events()
            self.assertEqual(index, 0, "Didn't query from first event")
            self.assertEqual(query, field_query, "Didn't unconvert query")

            index, query = events_query.query_multiple_events()
            self.assertEqual(index, 0, "Didn't query from first event")
            self.assertEqual(query, field_query, "Didn't unconvert query")

        for event_index, text in enumerate(base_queries):
            events_text = 'events[{}].{}'.format(event_index, text)
            field_query = parse_expression(text)  # type: Field
            events_query = parse_expression(events_text)  # type: Field
            index, query = events_query.query_multiple_events()
            self.assertEqual(index, event_index, "Didn't query from {} event".format(event_index))
            self.assertEqual(query, field_query, "Didn't unconvert query")

    def test_parse_with_preprocessor(self):
        """Test that preprocessor works with the parser."""
        preprocessor = get_preprocessor("""
        const ABC = 123
        const DEF = 456
        const GHI = 123

        macro COMPARE_TWO(a, b)  a == b
        macro GET_TRUE(a)   COMPARE_TWO(a, a)
        """)

        def p(text):
            return parse_expression(text, preprocessor=preprocessor)

        self.assertEqual(p('ABC'), Number(123))
        self.assertEqual(p('COMPARE_TWO(some_field, "abc.exe")'), p('some_field == "abc.exe"'))
        self.assertEqual(p('COMPARE_TWO(105, 105)'), Boolean(True))
        self.assertEqual(p('GET_TRUE(100)'), Boolean(True))

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

    def test_comments(self):
        """Test that comments are valid syntax but stripped from AST."""
        match = parse_query("process where pid=4 and ppid=0")

        query = parse_query("""process where pid = 4 /* multi\nline\ncomment */ and ppid=0""")
        self.assertEqual(match, query)

        query = parse_query("""process where pid = 4 // something \n and ppid=0""")
        self.assertEqual(match, query)

        query = parse_query("""process where pid
            = 4 and ppid=0
        """)
        self.assertEqual(match, query)

        query = parse_query("""process where
            // test
            //
            //line
            //comments
            pid = 4 and ppid = 0
        """)
        self.assertEqual(match, query)

        match = parse_expression("true")
        query = parse_expression("true // something else \r\n /* test\r\n something \r\n*/")
        self.assertEqual(match, query)

        commented = parse_definitions("macro test() pid = 4 and /* comment */ ppid = 0")
        macro = parse_definitions("macro test() pid = 4 and ppid = 0")
        self.assertEqual(commented, macro)

    def test_invalid_comments(self):
        """Test that invalid/overlapping comments fail."""
        query_text = "process where /* something */ else */ true"
        self.assertRaises(ParseError, parse_query, query_text)

        # Test nested comments (not supported)
        query_text = "process where /* outer /* nested */ outer */ true"
        self.assertRaises(ParseError, parse_query, query_text)

        query_text = "process where // true"
        self.assertRaises(ParseError, parse_query, query_text)
