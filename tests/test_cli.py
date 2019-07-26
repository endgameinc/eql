"""Test case for Command-Line entry point to EQL."""
import io
import json
import os
import uuid
import unittest

import mock

from eql.errors import EqlSchemaError
from eql.loader import save_analytics
from eql.main import main, BANNER, shell_main
from eql.parser import parse_analytics
from eql.schema import Schema
from eql.tests import TestEngine, EVENTS_FILE
from eql.table import Table
from eql.utils import save_dump, to_unicode

build_analytics = parse_analytics([
    {'query': "process where a == b", 'metadata': {'id': str(uuid.uuid4())}},
    {'query': "sequence [process where a == b] [file where x == y]", 'metadata': {'id': str(uuid.uuid4())}},
    {'query': "join by unique_pid [network where true] [dns where true]", 'metadata': {'id': str(uuid.uuid4())}},
])


def stdin_patch():
    """Patch stdin with a temporary stream."""
    return io.StringIO(u'\n'.join([json.dumps(event.data) for event in TestEngine.get_events()]))


class TestEqlCommand(unittest.TestCase):
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
        schema = {'magic': {"expected_field": "string"}}

        target_file = os.path.abspath('analytics-saved.tmp.json')
        analytics_file = os.path.abspath('analytics.tmp.json')
        config_file = os.path.abspath('config.tmp.json')

        if os.path.exists(target_file):
            os.remove(target_file)

        analytics = parse_analytics([
            {'query': "magic where actual_field = true", 'metadata': {'id': str(uuid.uuid4())}},
        ])
        save_analytics(analytics, analytics_file)
        save_dump({'schema': {"events": schema}, "allow_any": False}, config_file)

        with self.assertRaises(EqlSchemaError):
            main(['build', analytics_file, target_file, '--config', config_file, '--analytics-only'])

        self.assertFalse(os.path.exists(target_file))

        os.remove(config_file)
        os.remove(analytics_file)

    def test_engine_schema_implied(self):
        """Test building an engine with a custom config."""
        schema = {'magic': {}}

        target_file = os.path.abspath('analytics-saved.tmp.json')
        analytics_file = os.path.abspath('analytics.tmp.json')

        with Schema(schema):
            analytics = parse_analytics([{'query': "magic where true", 'metadata': {'id': str(uuid.uuid4())}}])
            save_analytics(analytics, analytics_file)

        main(['build', analytics_file, target_file])

        os.remove(analytics_file)
        os.remove(target_file)

    @mock.patch('eql.PythonEngine.print_event')
    @mock.patch('sys.stdin', new=stdin_patch())
    def test_query_eql_stdin(self, mock_print_event):
        """Stream stdin to the EQL command."""
        query = "process where true | head 8 | tail 1"
        main(['query', query])

        expected = [8]
        actual_event_ids = [args[0][0].data['serial_event_id'] for args in mock_print_event.call_args_list]
        self.assertEqual(expected, actual_event_ids, "Event IDs didn't match expected.")

    @mock.patch('eql.PythonEngine.print_event')
    @mock.patch('sys.stdin', new=stdin_patch())
    def test_implied_any(self, mock_print_event):
        """Stream stdin to the EQL command."""
        query = "true | unique event_type_full"
        main(['query', query])

        expected = [1, 55, 57, 63, 75304]
        actual_event_ids = [args[0][0].data['serial_event_id'] for args in mock_print_event.call_args_list]
        self.assertEqual(expected, actual_event_ids, "Event IDs didn't match expected.")

    @mock.patch('eql.PythonEngine.print_event')
    @mock.patch('sys.stdin', new=stdin_patch())
    def test_implied_base(self, mock_print_event):
        """Stream stdin to the EQL command."""
        query = "| unique event_type_full"
        main(['query', query])

        expected = [1, 55, 57, 63, 75304]
        actual_event_ids = [args[0][0].data['serial_event_id'] for args in mock_print_event.call_args_list]
        self.assertEqual(expected, actual_event_ids, "Event IDs didn't match expected.")

    @mock.patch('eql.PythonEngine.print_event')
    def test_query_eql_json(self, mock_print_event):
        """Test file I/O with EQL."""
        query = "process where true | head 8 | tail 1"
        main(['query', query, '-f', TestEngine.events_file])

        expected = [8]
        actual_event_ids = [args[0][0].data['serial_event_id'] for args in mock_print_event.call_args_list]
        self.assertEqual(expected, actual_event_ids, "Event IDs didn't match expected.")

    @mock.patch('eql.PythonEngine.print_event')
    def test_query_eql_jsonl(self, mock_print_event):
        """Test file I/O with EQL."""
        query = "process where true | head 8 | tail 1"
        main(['query', query, '-f', TestEngine.events_file])

        expected = [8]
        actual_event_ids = [args[0][0].data['serial_event_id'] for args in mock_print_event.call_args_list]
        self.assertEqual(expected, actual_event_ids, "Event IDs didn't match expected.")

    # TODO: Fix this test so it actually works
    def _test_interactive_shell(self):
        """Test that commands can be executed via the interactive shell."""
        class Arguments(object):
            config = None
            file = None

        actual_stdin = io.StringIO(to_unicode("\n".join([
            "input %s" % EVENTS_FILE,
            "table process_path parent_process_path",
            "search\nprocess where serial_event_id in (32, 33);",
        ])))

        expected_stdout_text = "\n".join([
            BANNER,
            "eql> input %s" % EVENTS_FILE,
            "Using file %s with %d events" % (EVENTS_FILE, len(TestEngine.get_events())),
            "eql> table process_path parent_process_path",
            "eql> search process where serial_event_id in (32, 33)",
            Table([
                ["C:\\Windows\\System32\\sppsvc.exe", "C:\\Windows\\System32\\services.exe"],
                ["C:\\Windows\\System32\\dwm.exe", "C:\\Windows\\System32\\svchost.exe"]
            ], names=["process_path", "parent_process_path"]).__unicode__()
        ])

        actual_stdout = []

        # Now actually run with redirected stdout and stdin
        with mock.patch('sys.stdin', new=actual_stdin):
            shell_main(Arguments())

        actual_stdout_lines = "\n".join(actual_stdout).splitlines()
        self.assertListEqual(actual_stdout_lines, expected_stdout_text.splitlines())
