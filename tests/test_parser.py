"""Test case."""
# -*- coding: utf-8 -*-
from __future__ import unicode_literals
import datetime
import sys
import traceback
import unittest

from eql.ast import *  # noqa: F403
from eql.errors import EqlSyntaxError, EqlSemanticError, EqlParseError
from eql.parser import (
    parse_query, parse_expression, parse_definitions, ignore_missing_functions, parse_field, parse_literal,
    extract_query_terms, keywords
)
from eql.walkers import DepthFirstWalker
from eql.pipes import *   # noqa: F403


class TestParser(unittest.TestCase):
    """Test EQL parsing."""

    def test_valid_expressions(self):
        """Test that expressions are parsed correctly."""
        valid = [
            "1 == 1",
            "false != (1 == 1)",
            'abc != "ghi"',
            "abc > 20",
            "startsWith(abc, 'abc')",
            "concat(a,b,c,d,)",
            "a in (1,2,3,4,)",
            "length(abc) < length(hij)",
            "length(concat(abc))",
            'abc == substring("abc", 1, 3)',
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
            '1 * 2 + 3 * 4 + 10 / 2',

            # opt-in with eql.parser.implied_booleans
            # 'abc == (1 and 2)',
            # 'abc == (def and 2)',

            'abc == (def and def)',
            'abc == (def and ghi)',
            '"\\b\\t\\r\\n\\f\\\\\\"\\\'"',
            '1 - -2',
            '1 + -2',
            '1 * (-2)',
            '3 * -length(file_path)',
        ]

        for query in valid:
            parse_expression(query)

    def test_parse_field(self):
        """Test that fields are parsed correctly."""
        self.assertEqual(parse_field("process_name  "), Field("process_name"))
        self.assertEqual(parse_field("TRUE  "), Field("TRUE"))
        self.assertEqual(parse_field("  data[0]"), Field("data", [0]))
        self.assertEqual(parse_field("data[0].nested.name"), Field("data", [0, "nested", "name"]))

        self.assertRaises(EqlParseError, parse_field, "  ")
        self.assertRaises(EqlParseError, parse_field, "100.5")
        self.assertRaises(EqlParseError, parse_field, "true")
        self.assertRaises(EqlParseError, parse_field, "and")
        self.assertRaises(EqlParseError, parse_field, "length(name) and path")

    def test_parse_literal(self):
        """Test that fields are parsed correctly."""
        self.assertEqual(parse_literal("true"), Boolean(True))
        self.assertEqual(parse_literal("null"), Null())
        self.assertEqual(parse_literal("  100.5  "), Number(100.5))
        self.assertEqual(parse_literal("true"), Boolean(True))
        self.assertEqual(parse_literal("'C:\\\\windows\\\\system32\\\\cmd.exe'"),
                         String("C:\\windows\\system32\\cmd.exe"))

        self.assertRaises(EqlParseError, parse_field, "and")
        self.assertRaises(EqlParseError, parse_literal, "process_name")
        self.assertRaises(EqlParseError, parse_literal, "length('abc')")
        self.assertRaises(EqlParseError, parse_literal, "True")

    def test_functions(self):
        """Test that functions are being parsed correctly."""
        # Make sure that functions are parsing all arguments
        with ignore_missing_functions:
            fn = parse_expression('somefunction( a and c, false, d or g) ')
        self.assertIsInstance(fn, FunctionCall)
        self.assertEqual(len(fn.arguments), 3)

    def test_invalid_expressions(self):
        """Test that expressions are parsed correctly."""
        invalid = [
            '',  # empty
            'a xor b',  # made up comparator
            'a ^ b',  # made up comparator
            'a b c d',  # missing syntax
            'def[]',  # no index
            'def[ghi]',  # index not a number
            'def[-1]',  # negative indexes not supported
            'someFunc().abc',  # invalid function
            'length().abc',  # can't index these
            '1.2.3',  # invalid number
            'a.1',
            '(field',  # unclosed paren
            '(field xx',  # unclosed paren and bad syntax
            'field[',  # unclosed bracket
            'field[0',  # unclosed bracket
            '(',
            ')',
            '()',  # nothing inside
            '',
            '"invalid"string"',
            'descendant of [event_type where true',
            '--100',
            '1000   100',
            '""    100',
            # literal values as fields and functions
            'true.100',
            'true()',
            'null.abc',
            'abc[0].null',
            # require escape slashes,
            '\\R',
            '\\W',
            # minimum of 1 argument
            'length()',
            'concat()',
        ]

        keywords = [
            'and', 'by', 'in', 'join', 'macro', 'not', 'of', 'or', 'sequence', 'until', 'where', 'with'
        ]

        for query in invalid:
            self.assertRaises(EqlParseError, parse_expression, query)

        for keyword in keywords:
            self.assertRaises(EqlSyntaxError, parse_expression, keyword)
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
            'process where _leadingUnderscore == 100',
            'network where 1 * 2 + 3 * 4 + 10 / 2 == 2 + 12 + 5',

            # now requires eql.parser.implied_booleans
            # 'file where (1 - -2)',
            # 'file where 1 + (-2)',
            # 'file where 1 * (-2)',
            # 'file where 3 * -length(file_path)',

            'network where a * b + c * d + e / f == g + h + i',
            'network where a * (b + c * d) + e / f == g + h + i',
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
            'sequence by pid with maxspan=200ms [process where process_name == "*" ] [file where file_path == "*"]',
            'sequence by pid with maxspan=1s [process where process_name == "*" ] [file where file_path == "*"]',
            'sequence by pid with maxspan=2h [process where process_name == "*"] [file where file_path == "*"]',
            'sequence by pid with maxspan=3d [process where process_name == "*"] [file where file_path == "*"]',
            'dns where pid == 100 | head 100 | tail 50 | unique pid',
            'network where pid == 100 | unique command_line | count',
            'security where user_domain == "endgame" | count user_name a b | tail 5',
            'process where 1==1 | count user_name, unique_pid, concat(field2,a,bc)',
            'process where 1==1 | unique user_name, concat(field2,a,bc), field2',
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
            'sequence [file where 1=1] by f1,f2 [process where 1=1] by f1,f2 until [process where 1=1] by f1,f2',
            'sequence by f [file where true] by a,b [process where true] by c,d until [process where 1=1] by e,f',
            # sequence with named params
            'sequence by unique_pid [process where true] [file where true] fork',
            'sequence by unique_pid [process where true] [file where true] fork=true',
            'sequence by unique_pid [process where true] [file where true] fork=false',
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

            except (EqlSyntaxError, EqlSemanticError):
                ex_type, ex, tb = sys.exc_info()
                traceback.print_exc()
                traceback.print_tb(tb)
                self.fail("Unable to parse query #{}: {}".format(i, text))

    def test_invalid_queries(self):
        """Test that invalid queries throw the proper error."""
        invalid = [
            '',  # empty
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

            # forks updated to stictly take true/false (true if not defined)
            'sequence by unique_pid [process where true] [file where true] fork=1',
            'sequence by unique_pid [process where true] [file where true] fork=0 [network where true]',
            'sequence by unique_pid [process where true] [file where true] fork=0',

            # time units made stricter, and floating points removed
            'sequence by pid with maxspan=2sec [process where process_name == "*" ] [file where file_path == "*"]',
            'sequence by pid with maxspan=200 [process where process_name == "*" ] [file where file_path == "*"]',
            'sequence by pid with maxspan=2seconds [process where process_name == "*" ] [file where file_path == "*"]',
            'sequence with maxspan=2.5m [process where x == x] by pid [file where file_path == "*"] by ppid',
            'sequence by pid with maxspan=2.0h [process where process_name == "*"] [file where file_path == "*"]',
            'sequence by pid with maxspan=2.0h [process where process_name == "*"] [file where file_path == "*"]',
            'sequence by pid with maxspan=1.0075d [process where process_name == "*"] [file where file_path == "*"]',
        ]
        for query in invalid:
            self.assertRaises(EqlParseError, parse_query, query)

    def test_backtick_fields(self):
        """Test that backticks are accepted with fields."""
        def parse_to(text, path):
            node = parse_expression(text)
            self.assertIsInstance(node, Field)
            self.assertEqual(node.full_path, path)

            # now render back as text and parse again
            node2 = parse_expression(node.render())
            self.assertEqual(node2, node)

        parse_to("`foo-bar-baz`", ["foo-bar-baz"])
        parse_to("`foo bar baz`", ["foo bar baz"])
        parse_to("`foo.bar.baz`", ["foo.bar.baz"])
        parse_to("`foo`.`bar-baz`", ["foo", "bar-baz"])
        parse_to("`foo.bar-baz`", ["foo.bar-baz"])
        parse_to("`💩`", ["💩"])

        parse_to("`foo`[0]", ["foo", 0])
        parse_to("`foo`[0].`bar`", ["foo", 0, "bar"])

        # keywords
        for keyword in keywords:
            parse_to("`{keyword}`".format(keyword=keyword), [keyword])
            parse_to("prefix.`{keyword}`".format(keyword=keyword), ["prefix", keyword])
            parse_to("`{keyword}`[0].suffix".format(keyword=keyword), [keyword, 0, "suffix"])

    def test_backtick_split_lines(self):
        """Confirm that backticks can't be split across lines."""
        with self.assertRaises(EqlSyntaxError):
            parse_expression("`abc \n def`")

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

    def test_float_time_unit(self):
        """Test that error messages are raised and formatted when time units are missing."""
        def error(query, message):
            with self.assertRaises(EqlSemanticError) as exc:
                parse_query(query)

            self.assertEqual(exc.exception.error_msg, message)

        error("sequence with maxspan=0.150s [foo where true] [bar where true]",
              "Only integer values allowed for maxspan. Did you mean 150ms?")

        error("sequence with maxspan=1.6h [foo where true] [bar where true]",
              "Only integer values allowed for maxspan.\nTry a more precise time unit: ms, s, m.")

        error("sequence with maxspan=0.5ms [foo where true] [bar where true]",
              "Only integer values allowed for maxspan.")

        error("sequence with maxspan=0.5zz [foo where true] [bar where true]",
              "Only integer values allowed for maxspan.")

    def test_invalid_comments(self):
        """Test that invalid/overlapping comments fail."""
        query_text = "process where /* something */ else */ true"
        self.assertRaises(EqlParseError, parse_query, query_text)

        # Test nested comments (not supported)
        query_text = "process where /* outer /* nested */ outer */ true"
        self.assertRaises(EqlParseError, parse_query, query_text)

        query_text = "process where // true"
        self.assertRaises(EqlParseError, parse_query, query_text)

    def test_invalid_time_unit(self):
        """Test that error messages are raised and formatted when time units are missing."""
        with self.assertRaisesRegex(EqlSemanticError, "Unknown time unit. Recognized units are: ms, s, m, h, d."):
            parse_query("sequence with maxspan=150 zz [foo where true] [bar where true]")

        with self.assertRaisesRegex(EqlSemanticError, "Unknown time unit. Recognized units are: ms, s, m, h, d."):
            parse_query("sequence with maxspan=150 hours [foo where true] [bar where true]")

    def test_method_syntax(self):
        """Test correct parsing and rendering of methods."""
        parse1 = parse_expression("(a and b):concat():length()")
        parse2 = parse_expression("a and b:concat():length() > 0")
        self.assertNotEqual(parse1, parse2)

        class Unmethodize(DepthFirstWalker):
            """Strip out the method metadata, so its rendered directly as a node."""

            def _walk_function_call(self, node):
                node.as_method = False
                return node

        without_method = Unmethodize().walk(parse1)
        expected = parse_expression("length(concat(a and b))")

        self.assertEqual(parse1, parse_expression("(a and b):concat():length()"))
        self.assertIsNot(parse1, without_method)
        self.assertEqual(without_method, expected)

    def test_missing_time_unit(self):
        """Test that error messages are raised and formatted when time units are missing."""
        with self.assertRaisesRegex(EqlSemanticError, "Missing time unit. Did you mean 150s?"):
            parse_query("sequence with maxspan=150 [foo where true] [bar where true]")

    def test_term_extraction(self):
        """Test that EQL terms are correctly extracted."""
        process_event = """
            process where process_name == "net.exe" and child of [
                network where destination_port == 443
            ]
        """
        file_event = "file where false"
        network_event = "   network   where\n\n\n\n   destination_address='1.2.3.4'\n\t  and destination_port == 8443"

        sequence_template = "sequence with maxspan=10m [{}] by field1, field2, [{}] by field2, field3 [{}] by f4, f5"
        join_template = "join [{}] by a [{}] by b [{}] by c until [dns where false] by d"

        # basic sequence with by
        terms = [process_event, network_event, file_event]
        stripped = [t.strip() for t in terms]
        sequence_extracted = extract_query_terms(sequence_template.format(*terms))
        self.assertListEqual(sequence_extracted, stripped)

        # sequence with by and pipes
        terms = [network_event, process_event, process_event]
        stripped = [t.strip() for t in terms]
        sequence_extracted = extract_query_terms(sequence_template.format(*terms) + "| head 100 | tail 10")
        self.assertListEqual(sequence_extracted, stripped)

        # join with by
        terms = [network_event, process_event, process_event]
        stripped = [t.strip() for t in terms]
        join_extracted = extract_query_terms(join_template.format(*terms))
        self.assertListEqual(join_extracted, stripped)

        # simple query without pipes
        simple_extracted = extract_query_terms(network_event)
        self.assertListEqual(simple_extracted, [network_event.strip()])

        # simple query with pipes
        simple_extracted = extract_query_terms(network_event + "| unique process_name, user_name\n\n| tail 10")
        self.assertListEqual(simple_extracted, [network_event.strip()])
