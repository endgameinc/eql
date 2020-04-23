"""Test case."""
import unittest

from eql.events import Event
from eql.errors import EqlSchemaError, EqlTypeMismatchError
from eql.parser import parse_query, strict_booleans, non_nullable_fields, allow_enum_fields
from eql.schema import Schema
from eql.types import TypeHint


STR = TypeHint.String.value
NUM = TypeHint.Numeric.value
BOOL = TypeHint.Boolean.value
MIXED = TypeHint.Unknown.value


class TestSchemaValidation(unittest.TestCase):
    """Tests for schema lookups and type system."""

    schema = {
        "process": {
            "command_line": STR,
            "process_name": STR,
            "pid": NUM,
            "elevated": BOOL

        },
        "file": {
            "file_path": STR,
            "file_name": STR,
            "process_name": NUM,
            "pid": NUM,
            "data": MIXED,
        },
        "complex": {
            "string_arr": [STR],
            "wideopen": {},
            "nested": {
                "arr": [MIXED],
                "double_nested": {
                    "nn": NUM, "triplenest": {"m": MIXED, "b": BOOL}
                },
                "num": NUM
            },
            "objarray": [{}],
        }
    }

    def test_valid_schema_event(self):
        """Test that schema errors are being raised separately."""
        valid = [
            'process where true',
            'file where true',
            'complex where true',
            'any where true',
            'generic where true'
        ]

        with Schema(self.schema, allow_generic=True, allow_any=True):
            for query in valid:
                parse_query(query)

    def test_invalid_schema_event(self):
        """Test that schema errors are being raised separately."""
        invalid = [
            'PROCESS where true',
            'network where true',
            'person where true',
            'generic where true',
            'any where true'
        ]

        with Schema(self.schema, allow_generic=False, allow_any=False):
            for query in invalid:
                self.assertRaises(EqlSchemaError, parse_query, query)

    def test_valid_schema_fields(self):
        """Test that schema errors are never raised for valid schema usage."""
        valid = [
            'process where process_name == "test" and command_line == "test" and pid > 0',
            'file where file_path == "abc" and data == 1',
            'file where file_path == "abc" and data == "fdata.exe"',
            'file where file_path == "abc" and data != null',
            'file where file_path == "abc" and length(data) > 0 | filter file_path == "abc"',
            'sequence [file where pid=1] [process where pid=2] | filter events[0].file_name = events[1].process_name',
            'sequence by pid [file where true] [process where true] ' +
            '| filter events[0].file_name = events[1].process_name',
            'join by pid [file where true] [process where true] | filter events[0].file_name = events[1].process_name',

            'join [file where true] by pid [process where true] by pid until [complex where false] by nested.num'
            '| filter events[0].file_name = events[1].process_name',

            'complex where string_arr[3] != null',
            'complex where wideopen.a.b[0].def == 1',
            'complex where length(nested.arr) > 0',
            'complex where nested.arr[0] == 1',
            'complex where nested.double_nested.nn == 5',
            'complex where nested.double_nested.triplenest != null',
            'complex where nested.double_nested.triplenest.m == 5',
            'complex where nested.  double_nested.triplenest.b != null',
        ]

        with Schema(self.schema):
            for query in valid:
                parse_query(query)

    def test_schema_enum_enabled(self):
        """Test that enum fields are converted to string comparisons."""
        with Schema(self.schema), allow_enum_fields:
            actual = parse_query("process where process_name.bash")
            expected = parse_query("process where process_name == 'bash'")
            self.assertEqual(actual, expected)

    def test_schema_enum_disabled(self):
        """Test that enum errors are raised when not explicitly enabled."""
        with Schema(self.schema):
            self.assertRaises(EqlSchemaError, parse_query, "process where process_name.bash")

    def test_invalid_schema_fields(self):
        """Test that schema errors are being raised separately."""
        invalid = [
            'process where not bad_field',
            'process where file_path',
            'file where command_line',
            'process where wideopen.a.b.c',
            'any where invalid_field',
            'complex where nested.  double_nested.b',
            'file where file_path == "abc" and data != null | unique missing_field == "abc"',
            'sequence [file where pid=1] [process where pid=2] | filter events[0].file_name = events[1].bad',

            'sequence [file where 1=1] by pid [process where 1=1] by pid until [complex where false] by pid'
            '| unique events[0].file_name = events[1].process_name',
        ]
        with Schema(self.schema):
            for query in invalid:
                with self.assertRaises(EqlSchemaError):
                    parse_query(query)

    def test_array_functions(self):
        """Test that array functions match array fields."""
        valid = [
            "complex where arrayContains(string_arr, 'thesearchstring')",
            "complex where arrayContains(string_arr, 'thesearchstring', 'anothersearchstring')",
            # this should pass until generics/templates are handled better
            "complex where arrayContains(string_arr, 1)",
            "complex where arrayContains(string_arr, 1, 2, 3)",
            "complex where arraySearch(string_arr, x, x == '*subs*')",
            "complex where arraySearch(objarray, x, x.key == 'k')",
            "complex where arraySearch(objarray, x, arraySearch(x, y, y.key == true))",
            "complex where arraySearch(nested.arr, x, x == '*subs*')",
            "complex where arrayContains(objarray, 1)",
            "complex where arrayContains(objarray, 1, 2, 3)",
        ]
        with Schema(self.schema):
            for query in valid:
                parse_query(query)

    def test_array_function_failures(self):
        """Test that array functions fail on nested objects or the wrong type."""
        valid = [
            "process where arrayContains(pid, 4)",
            "process where arraySearch(pid, x, true)",
            "complex where arraySearch(objarray, '*subs*')",
        ]
        with Schema(self.schema):
            for query in valid:
                self.assertRaises(EqlTypeMismatchError, parse_query, query)

    def test_strict_schema_failures(self):
        """Check that fields can't be compared to null under strict schemas."""
        queries = [
            "process where command_line != null",
            "process where elevated != null",
            "process where pid != null",
            "process where pid > null",

            # constants also aren't comparable to null
            "process where 5 > null",
            "process where 5 != null",

            "process where (pid == 0 or process_name == 'foo') == null",

            "process where indexOf(null, null)",

            "process where (process_name in ('net.exe')) != null",
            "process where (pid > 0) != null",
            "process where (pid > 0) != null",
            "process where substring(process_name, 1) == null",
        ]

        with strict_booleans, non_nullable_fields, Schema(self.schema):
            for query in queries:
                self.assertRaises(EqlTypeMismatchError, parse_query, query)

    def test_strict_schema_success(self):
        """Check that fields can't be compared to null under strict schemas."""
        queries = [
            "process where command_line != 'abc.exe'",
            "process where elevated != true",
            "process where not elevated",

            # functions that may return null, even with non-null inputs can be compared
            "process where indexOf(process_name, 'foo') != null",

            # since indexOf can be passed into other functions, any can be null
            "process where substring(process_name, indexOf(process_name, 'foo')) == null",
        ]

        with strict_booleans, non_nullable_fields, Schema(self.schema):
            for query in queries:
                parse_query(query)

    def test_count_schemas(self):
        """Test that schemas are updated with counts in pipes."""
        queries = [
            "process where true | count | filter key == 'total' and percent < 0.5 and count > 0",
            "process where true | unique_count process_name | filter count > 5 and process_name == '*.exe'",
            "sequence[file where true][process where true] | unique_count events[0].process_name" +
            " | filter count > 5 and events[1].elevated",
        ]

        with Schema(self.schema):
            for query in queries:
                parse_query(query)

    def test_count_schema_failures(self):
        """Test that schemas aren't overly updated with counts in pipes."""
        queries = [
            "process where true | count | filter key == 'total' and percent < 0.5 and count > 0 and elevated",
            "process where true | unique_count process_name | filter key == '*.abc'" +
            " and count > 5 and process_name == '*.exe'",
            "sequence[file where true][process where true] " +
            " | unique_count events[0].process_name" +
            " | filter count > 5 and events[0].elevated",
        ]

        with Schema(self.schema):
            for query in queries:
                self.assertRaises(EqlSchemaError, parse_query, query)

    def test_merge_schema(self):
        """Merge two schemas together."""
        a = Schema({"process": {"a": "string", "b": "number", "c": {}}})
        b = Schema({"process": {"c": "mixed"}, "file": {"path": "string"}})

        # Test that schemas prefer keys from the first
        c = a.merge(b)
        self.assertDictEqual(c.schema, {"process": {"a": "string", "b": "number", "c": {}}, "file": {"path": "string"}})

    def test_learn_schema(self):
        """Test that schemas can be learned from a set of data."""
        data = [
            {"event_type": "process", "a": {"b": 1, "c": 2}, "d": "e"},
            {"event_type": "file", "a": "b", "cd": ["ef", 123]},
            {"event_type": "process", "a": {"b": 1, "c": "e"}},
        ]
        event_schema = {
            "process": {"a": {"b": "number", "c": "mixed"}, "d": "string", "event_type": "string"},
            "file": {"a": "string", "cd": ["number", "string"], "event_type": "string"},
        }
        schema = Schema.learn(Event.from_data(d) for d in data)
        self.assertDictEqual(schema.schema, event_schema)
        self.assertFalse(schema.allow_generic)

        schema = Schema.learn([Event("generic", 0, {"a": "b"})])
        self.assertDictEqual(schema.schema, {"generic": {"a": "string"}})
        self.assertTrue(schema.allow_generic)
