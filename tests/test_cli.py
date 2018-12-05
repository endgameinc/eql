"""Test case for Command-Line entry point to EQL."""
import io
import json
import os
import uuid

import mock

from .base import TestEngine
from eql.errors import SchemaError
from eql.loader import save_analytics
from eql.main import main
from eql.parser import parse_analytics
from eql.schema import use_schema
from eql.utils import save_dump


build_analytics = parse_analytics([
    {'query': "process where a == b", 'metadata': {'id': str(uuid.uuid4())}},
    {'query': "sequence [process where a == b] [file where x == y]", 'metadata': {'id': str(uuid.uuid4())}},
    {'query': "join by unique_pid [network where true] [dns where true]", 'metadata': {'id': str(uuid.uuid4())}},
])


def stdin_patch():
    """Patch stdin with a temporary stream."""
    return io.StringIO(u'\n'.join([json.dumps(event.data) for event in TestEngine.get_events()]))


class TestEqlCommand(TestEngine):
    """Test EQL command line parsing and functionality."""

    @mock.patch('sys.stdout')
    @mock.patch('sys.stderr')
    def test_incomplete_args(self, mock_stdout, mock_stderr):
        """Check that missing arguments cause failures."""
        # too few arguments
        self.assertRaises(SystemExit, main, ['build'])

        # too few arguments
        self.assertRaises(SystemExit, main, ['query'])

        # invalid choice
        self.assertRaises(SystemExit, main, ['bad'])

    @mock.patch('argparse.ArgumentParser.error')
    def test_engine_config(self, mock_error):
        """Test building an engine with a custom config."""
        schema = {'event_types': {'magic': 543212345}}

        target_file = os.path.abspath('analytics-saved.tmp.json')
        analytics_file = os.path.abspath('analytics.tmp.json')
        config_file = os.path.abspath('config.tmp.json')

        with use_schema(schema):
            analytics = parse_analytics([{'query': "magic where true", 'metadata': {'id': str(uuid.uuid4())}}])
            save_analytics(analytics, analytics_file)
            with open(analytics_file, 'r') as f:
                expected_contents = f.read()

        save_dump({'schema': schema}, config_file)

        main(['build', analytics_file, target_file, '--config', config_file, '--analytics-only'])

        with open(target_file, 'r') as f:
            actual_contents = f.read()

        self.assertEqual(actual_contents, expected_contents)

        os.remove(config_file)
        os.remove(target_file)
        os.remove(analytics_file)

    def test_engine_schema_failure(self):
        """Test building an engine with a custom config."""
        schema = {'event_types': {'magic': 543212345}}

        target_file = os.path.abspath('analytics-saved.tmp.json')
        analytics_file = os.path.abspath('analytics.tmp.json')

        with use_schema(schema):
            analytics = parse_analytics([{'query': "magic where true", 'metadata': {'id': str(uuid.uuid4())}}])
            save_analytics(analytics, analytics_file)

        with self.assertRaises(SchemaError):
            main(['build', analytics_file, target_file])

        os.remove(analytics_file)

    @mock.patch('eql.engines.native.PythonEngine.print_event')
    @mock.patch('sys.stdin', new=stdin_patch())
    def test_query_eql_stdin(self, mock_print_event):
        """Stream stdin to the EQL command."""
        query = "process where true | head 8 | tail 1"
        main(['query', query])

        expected = [8]
        actual_event_ids = [args[0][0].data['serial_event_id'] for args in mock_print_event.call_args_list]
        self.assertEqual(expected, actual_event_ids, "Event IDs didn't match expected.")

    @mock.patch('eql.engines.native.PythonEngine.print_event')
    @mock.patch('sys.stdin', new=stdin_patch())
    def test_implied_any(self, mock_print_event):
        """Stream stdin to the EQL command."""
        query = "true | unique event_type_full"
        main(['query', query])

        expected = [1, 55, 57, 63, 75304]
        actual_event_ids = [args[0][0].data['serial_event_id'] for args in mock_print_event.call_args_list]
        self.assertEqual(expected, actual_event_ids, "Event IDs didn't match expected.")

    @mock.patch('eql.engines.native.PythonEngine.print_event')
    @mock.patch('sys.stdin', new=stdin_patch())
    def test_implied_base(self, mock_print_event):
        """Stream stdin to the EQL command."""
        query = "| unique event_type_full"
        main(['query', query])

        expected = [1, 55, 57, 63, 75304]
        actual_event_ids = [args[0][0].data['serial_event_id'] for args in mock_print_event.call_args_list]
        self.assertEqual(expected, actual_event_ids, "Event IDs didn't match expected.")

    @mock.patch('eql.engines.native.PythonEngine.print_event')
    def test_query_eql_json(self, mock_print_event):
        """Test file I/O with EQL."""
        query = "process where true | head 8 | tail 1"
        main(['query', query, '-f', self.EVENTS_FILE])

        expected = [8]
        actual_event_ids = [args[0][0].data['serial_event_id'] for args in mock_print_event.call_args_list]
        self.assertEqual(expected, actual_event_ids, "Event IDs didn't match expected.")

    @mock.patch('eql.engines.native.PythonEngine.print_event')
    def test_query_eql_jsonl(self, mock_print_event):
        """Test file I/O with EQL."""
        query = "process where true | head 8 | tail 1"
        main(['query', query, '-f', self.EVENTS_FILE])

        expected = [8]
        actual_event_ids = [args[0][0].data['serial_event_id'] for args in mock_print_event.call_args_list]
        self.assertEqual(expected, actual_event_ids, "Event IDs didn't match expected.")
