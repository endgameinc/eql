"""Test case for utility functions."""
import json
import os
import unittest

import eql.utils


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
        parsed = list(eql.utils.stream_json(jsonl.splitlines()))
        self.assertEqual(parsed, example, "JSON lines didn't stream properly.")
