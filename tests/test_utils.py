"""Test case for utility functions."""
import json
import os
import unittest

import eql.utils
from eql.errors import EqlParseError
from eql.parser import parse_query, parse_expression, parse_analytic
from eql.utils import is_stateful, match_kv, get_output_types, get_query_type, uses_ancestry, get_required_event_types


class TestUtils(unittest.TestCase):
    """Test Utility Functions."""

    def test_json_files(self):
        """Test load and save functionality for json."""
        filename = 'tmp.json'
        data = {'key1': 'value1', 'key2': [{}], 'key3': [1, 2, 3]}

        eql.utils.save_dump(data, filename)
        loaded = eql.utils.load_dump(filename)
        self.assertEqual(loaded, data)
        os.remove(filename)

    def test_yml_files(self):
        """Test load and save functionality for yml."""
        filename = 'tmp.yml'
        data = {'key1': 'value1', 'key2': [{}], 'key3': [1, 2, 3]}

        eql.utils.save_dump(data, filename)
        loaded = eql.utils.load_dump(filename)
        self.assertEqual(loaded, data)
        os.remove(filename)

        filename = 'tmp.yaml'
        eql.utils.save_dump(data, filename)
        loaded = eql.utils.load_dump(filename)
        self.assertEqual(loaded, data)
        os.remove(filename)

    def test_stream_jsonl(self):
        """Check streaming of jsonlines."""
        example = [{'a': '1'}, {'b': '2'}, {'c': '3'}]
        jsonl = '\n'.join(json.dumps(item) for item in example)
        parsed = list(eql.utils.stream_json_lines(jsonl.splitlines()))
        self.assertEqual(parsed, example, "JSON lines didn't stream properly.")

    def test_stateful_checks(self):
        """Check that :func:`~utils.is_stateful` is identifying stateful queries."""
        stateful_queries = [
            "sequence [process where process_name='net.exe']  [process where process_name='net.exe']",
            "join [process where process_name='net.exe']  [process where process_name='net.exe']",
            "file where file_name='*.txt' and descendant of [process where pid=4]",
            "file where child of [process where pid=4]",
            "registry where event of [process where pid=4]",
            "process where true | unique_count process_name | filter count < 5",
            "any where true | count user_name",
        ]

        for query in stateful_queries:
            ast = parse_query(query)
            self.assertTrue(is_stateful(ast), "{} was not recognized as stateful".format(query))

    def test_stateless_checks(self):
        """Check that :func:`~utils.is_stateful` is identifying stateless queries."""
        stateless_queries = [
            "process where true | filter command_line='* https://*' | tail 10",
            "process where user_name='system' | unique parent_process_name | head 500",
            "file where file_name='*.txt' and (process_name='cmd.exe' or parent_process_name='net.exe')",
            "registry where length(user_name) == 500",
            "network where string(destination_port) == '500' | unique process_name",
        ]

        for query in stateless_queries:
            ast = parse_query(query)
            self.assertFalse(is_stateful(ast), "{} was not recognized as stateless".format(query))

    def test_match_kv(self):
        """Check that :func:~utils.match_kv~ returns the expected EQL expressions."""
        def assert_kv_match(condition_dict, condition_text, *args):
            """Helper function for validation."""
            condition_node = match_kv(condition_dict)
            parsed_node = parse_expression(condition_text)
            self.assertEqual(condition_node.render(), parsed_node.render(), *args)

        assert_kv_match({"name": "net.exe"},
                        r"name == 'net.exe'",
                        "Simple KV match")

        assert_kv_match({"name": ["net.exe"]},
                        r"name == 'net.exe'",
                        "Single list match")

        assert_kv_match({"path": ["C:\\windows\\system32\\net.exe"]},
                        r"path == 'C:\\windows\\system32\\net.exe'",
                        "String escaping")

        assert_kv_match({"path": ["C:\\windows\\system32\\net*.exe", "C:\\windows\\*\\cmd.exe"]},
                        r"wildcard(path, 'C:\\windows\\system32\\net*.exe', 'C:\\windows\\*\\cmd.exe')",
                        "Multiple wildcards")

        assert_kv_match({"nested[0].name.test": ["net.exe"]},
                        r"nested[0].name.test == 'net.exe'",
                        "Nested field check")

        assert_kv_match({"name": ["net.exe", "net1.exe"]},
                        r"name in ('net.exe', 'net1.exe')",
                        "Multiple values in list")

        assert_kv_match({"name": ["net.exe", "net1.exe"], "pid": 4},
                        r"name in ('net.exe', 'net1.exe') and pid == 4",
                        "Multiple fields checked")

        assert_kv_match({"completed": True, "delta": [8.2, 8.4]},
                        r"completed == true and delta in (8.2, 8.4)",
                        "Booleans and floats")

        assert_kv_match({"name": ["net.exe", "net1.exe", "cmd*.exe"], "pid": [4]},
                        r"(name in ('net.exe', 'net1.exe') or name == 'cmd*.exe') and pid == 4",
                        "Complex query")

        # Test for nested fields
        assert_kv_match({"events[0].process_path": "c:\\windows\\explorer.exe",
                         "events[1].file_name": "test.docx"},
                        r"events[0].process_path == 'c:\\windows\\explorer.exe' and events[1].file_name == 'test.docx'",
                        "Nested fields")

        assert_kv_match({"triggering_fact_array[0].data_buffer.process_path": "c:\\windows\\explorer.exe"},
                        r"triggering_fact_array[0].data_buffer.process_path == 'c:\\windows\\explorer.exe'")

        assert_kv_match({}, "true", "Empty dict")
        assert_kv_match({"empty": []}, "false", "Empty list of values")

    def test_match_kv_errors(self):
        """Test that KV matching raises errors when expected."""
        self.assertRaises(EqlParseError, match_kv, {"invalid^field&syntax": "abc"})
        self.assertRaises(EqlParseError, match_kv, {"100": "invalid field"})

        # Test that the parameters are validated
        self.assertRaises(TypeError, match_kv, {"process_name": ["a", tuple()]})
        self.assertRaises(TypeError, match_kv, [])
        self.assertRaises(TypeError, match_kv, True)
        self.assertRaises(TypeError, match_kv, 1)

    def test_query_type(self):
        """Check eql.utils.get_query_type."""
        self.assertEqual(get_query_type(parse_query("any where true")), "event")
        self.assertEqual(get_query_type(parse_query("process where true")), "event")
        self.assertEqual(get_query_type(parse_query("sequence [process where true] [network where true]")), "sequence")
        self.assertEqual(get_query_type(parse_query("join [process where true] [network where true]")), "join")

    def test_required_event_types(self):
        """Test that ancestry checks are detected."""
        self.assertSetEqual(get_required_event_types(parse_query("file where true")), {"file"})

        self.assertSetEqual(get_required_event_types(parse_query("any where event of [process where true]")),
                            {"any", "process"})
        self.assertSetEqual(get_required_event_types(parse_query("any where descendant of [process where true]")),
                            {"any", "process"})

        self.assertSetEqual(get_required_event_types(parse_query("""
        sequence
            [file where true]
            [process where true]
            [network where true]
        """)), {"file", "process", "network"})

        self.assertSetEqual(get_required_event_types(parse_query("""
        join
            [file where true]
            [process where true]
            [network where true]
        """)), {"file", "process", "network"})

        self.assertSetEqual(get_required_event_types(parse_query("""
        file where descendant of [
            dns where child of
                [registry where true]]
        """)), {"file", "dns", "registry"})

        self.assertSetEqual(get_required_event_types(parse_query("""
        sequence
            [file where descendant of [dns where child of [registry where true]]]
            [process where true]
            [network where true]
        """)), {"file", "dns", "network", "process", "registry"})

    def test_uses_ancestry(self):
        """Test that ancestry checks are detected."""
        self.assertFalse(uses_ancestry(parse_query("any where true")))
        self.assertTrue(uses_ancestry(parse_query("any where child of [any where true]")))
        self.assertTrue(uses_ancestry(parse_query("any where descendant of [any where true]")))
        self.assertTrue(uses_ancestry(parse_query("any where event of [any where true]")))

        self.assertFalse(uses_ancestry(parse_query("sequence [process where true] [network where true]")))
        self.assertTrue(uses_ancestry(parse_query("""
        sequence
            [process where child of [file where true]]
            [network where true]
        """)))
        self.assertTrue(uses_ancestry(parse_query("""
        join
            [process where event of [file where true]]
            [network where true]
        """)))
        self.assertTrue(uses_ancestry(parse_query("""
        join
            [process where descendant of [file where true]]
            [network where true]
        """)))

    def test_output_types(self):
        """Test that output types are correctly returned from eql.utils.get_output_types."""
        query_ast = parse_query("process where true")
        self.assertEqual(get_output_types(query_ast), ["process"])

        query_ast = parse_analytic({"query": "process where descendant of [file where true]"})
        self.assertEqual(get_output_types(query_ast), ["process"])

        query_ast = parse_query("file where true | unique pid | head 1")
        self.assertEqual(get_output_types(query_ast), ["file"])

        query_ast = parse_query("file where true | unique_count file_path")
        self.assertEqual(get_output_types(query_ast), ["file"])

        query_ast = parse_query("any where true | unique_count file_path")
        self.assertEqual(get_output_types(query_ast), ["any"])

        query_ast = parse_query("file where true | count")
        self.assertEqual(get_output_types(query_ast), ["generic"])

        query_ast = parse_query("file where true | count process_name")
        self.assertEqual(get_output_types(query_ast), ["generic"])

        query_ast = parse_query("""
        sequence
            [registry where true]
            [file where true]
            [process where true]
            [process where true]
            [process where true]
            [network where true]
        """)
        self.assertEqual(get_output_types(query_ast), ["registry", "file", "process", "process", "process", "network"])

        query_ast = parse_query("""
        sequence
            [registry where true]
            [file where true]
            [process where true]
            [process where true]
            [process where true]
            [network where true]
        | count event_type
        | head 5
        """)
        self.assertEqual(get_output_types(query_ast), ["generic"])

        query_ast = parse_query("""
        sequence
            [registry where true]
            [file where true]
            [process where true]
            [process where true]
            [process where true]
            [network where true]
        | unique events[2].event_type
        | sort events[1].file_size
        | head 5
        | filter events[4].process_name == 'test.exe'
        """)
        self.assertEqual(get_output_types(query_ast), ["registry", "file", "process", "process", "process", "network"])
