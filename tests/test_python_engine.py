"""Test Python Engine for EQL."""
import random
import uuid
from collections import defaultdict

from eql.engines.base import Event, AnalyticOutput
from eql.engines.build import get_reducer, get_engine, get_post_processor
from eql.engines.native import PythonEngine
from eql.parser import parse_query, parse_analytic
from eql.schema import EVENT_TYPE_GENERIC
from .base import TestEngine


class TestPythonEngine(TestEngine):
    """Test correctness of lambda generation for python engine."""

    engine_name = "python"

    def test_event_load(self):
        """Test that events can be loaded from valid data buffers or full json."""
        event_type = 'process'
        event_time = 131509374020000000
        event_data = {
            "command_line": "\\??\\C:\\Windows\\system32\\conhost.exe",
            "event_type_full": "process_event",
            "parent_process_name": "csrss.exe",
            "parent_process_path": "C:\\Windows\\System32\\csrss.exe",
            "pid": 3080,
            "ppid": 372,
            "process_name": "conhost.exe",
            "process_path": "C:\\Windows\\System32\\conhost.exe",
            "serial_event_id": 49,
            "timestamp": 131509374020000000,
            "user_domain": "vagrant",
            "user_name": "vagrant",
        }
        expected_event = Event(event_type, event_time, event_data)
        from_full_event = Event.from_data({
            'event_timestamp': event_time,
            'event_type': 4,
            'data_buffer': event_data
        })
        from_buffer = Event.from_data(event_data)
        self.assertEqual(from_full_event, expected_event, "Full event didn't load properly")
        self.assertEqual(from_buffer, expected_event, "Event buffer didn't load properly")

    def test_nested_data(self):
        """Test that highly structured is also searchable."""
        event_1 = {'top': [{'middle': {'abc': 0}}, {'middle2': ['def', 'ghi']}]}
        event_2 = {'top': [{'middle': {'abc': 123}}, {'middle2': ['tuv', 'wxyz']}]}
        events = [Event(EVENT_TYPE_GENERIC, 1, event_1), Event(EVENT_TYPE_GENERIC, 2, event_2)]

        query = parse_query('generic where top[0].middle.abc == 123')
        results = self.get_output(queries=[query], events=events, config={'flatten': True})
        self.assertEqual(len(results), 1, "Missing or extra results")
        self.assertEqual(results[0].data, event_2, "Failed to match on correct event")

    def test_engine_load(self):
        """Check that various queries can be converted and loaded into the python engine."""
        engine = PythonEngine()
        engine.add_custom_function('myFn', lambda x, y, z: 100)
        queries = [
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
            'join \t\t\t[process where process_name == "*"] [  file where file_path == "*"\n]',
            'join by pid [process where name == "*"] [file where path == "*"] until [process where opcode == 2]',
            'sequence [process where name == "*"] [file where path == "*"] until [process where opcode == 2]',
            'sequence by pid [process where name == "*"] [file where path == "*"] until [process where opcode == 2]',
            'join [process where process_name == "*"] by process_path [file where file_path == "*"] by image_path',
            'sequence [process where process_name == "*"] by process_path [file where file_path == "*"] by image_path',
            'sequence by pid [process where process_name == "*"] [file where file_path == "*"]',
            'sequence by pid with maxspan=2s [process where process_name == "*" ] [file where file_path == "*"]',
            'sequence by pid with maxspan=2sec [process where process_name == "*" ] [file where file_path == "*"]',
            'sequence by pid with maxspan=2seconds [process where process_name == "*" ] [file where file_path == "*"]',
            'sequence with maxspan=2.5m [process where x == x] by pid [file where file_path == "*"] by ppid',
            'sequence by pid with maxspan=2.0h [process where process_name == "*"] [file where file_path == "*"]',
            'sequence by pid with maxspan=2.0h [process where process_name == "*"] [file where file_path == "*"]',
            'sequence by pid with maxspan=1.0075d [process where process_name == "*"] [file where file_path == "*"]',
            'process where descendant of [process where process_name == "lsass.exe"] and process_name == "cmd.exe"',
            'dns where pid == 100 | head 100 | tail 50 | unique pid',
            'network where pid == 100 | unique command_line | count',
            'security where user_domain == "endgame" | count user_name | tail 5',
            'process where 1==1 | count user_name, unique_pid, myFn(field2,a,bc)',
            'process where 1==1 | unique user_name, myFn(field2,a,bc), field2',
            'process where true',
            'any where topField.subField[100].subsubField == 0',
            'process where true | filter true',
            'process where 1==1 | filter abc == def',
            'process where 1==1 | filter abc == def and 1 != 2',
            'process where 1==1 | count process_name | filter percent > 0.5',
            'process where a > 100000000000000000000000000000000',
        ]
        for query in queries:
            # Make sure every query can be converted without raising any exceptions
            parsed_query = parse_query(query)
            engine.add_query(parsed_query)

            # Also try to load it as an analytic
            parsed_analytic = parse_analytic({'metadata': {'id': uuid.uuid4()}, 'query': query})
            engine.add_analytic(parsed_analytic)

    def test_raises_errors(self):
        """Confirm that exceptions are raised when expected."""
        queries = [
            # ('process where bad_field.sub_field == 100', AttributeError),
            ('process where length(0)', TypeError),
            # ('file where file_name.abc', AttributeError),
            # ('file where pid.something', AttributeError),
            ('registry where invalidFunction(pid, ppid)', KeyError),
        ]

        # Make sure that these all work as expected queries
        for query, expected_error in queries:
            parsed_query = parse_query(query)
            self.assertRaises(expected_error, self.get_output, queries=[parsed_query])

    def test_query_output(self):
        """Confirm that the known queries and data return expected output."""
        queries = self.get_example_queries()
        config = {'flatten': True, 'data_source': 'endgame'}
        # Make sure that these all work as expected queries
        for query_check in queries:
            query = query_check['query']
            expected_ids = query_check['expected_event_ids']
            analytic = query_check['analytic']
            output = self.get_output(queries=[analytic.query], config=config)
            actual_ids = [event.data['serial_event_id'] for event in output]
            self.validate_results(actual_ids, expected_ids, query)

    def get_output(self, queries=None, analytics=None, events=None, config=None):
        """Run a query over a data set and get all of the output events."""
        events = events or self.get_events()
        engine = PythonEngine(config)
        engine.add_custom_function('echo', self._custom_echo)
        engine.add_custom_function('reverse', self._custom_reverse)

        results = []  # type: list[Event]
        engine.add_output_hook(results.append)
        engine.add_queries(queries or [])
        engine.add_analytics(analytics or [])
        engine.stream_events(events)
        return results

    def test_map_reduce_queries(self):
        """Test map reduce functionality of python engines."""
        input_events = defaultdict(list)
        host_results = []

        for i, host in enumerate("abcdefghijklmnop"):
            events = []
            for event_number in range(10):
                data = {'number': event_number, 'a':  host + '-a-' + str(event_number), 'b': -event_number}
                events.append(Event.from_data(data))
            input_events[host] = events

        query_text = 'generic where true | sort a | head 5 | sort b'
        query = parse_query(query_text)
        host_engine = get_engine(query, {'flatten': True})

        # Map across multiple 'hosts'
        for hostname, host_events in input_events.items():
            for event in host_engine(host_events):
                event.data['hostname'] = hostname
                host_results.append(event)

        # Reduce across multiple 'hosts'
        reducer = get_reducer(query)
        reduced_results = reducer(host_results)

        expected_a = ['a-a-{}'.format(value) for value in range(10)][:5][::-1]
        actual_a = [event.data['a'] for result in reduced_results for event in result.events]
        self.validate_results(actual_a, expected_a, query_text)

    def test_aggregate_total_counts(self):
        """Test that total counts are aggregated correctly."""
        hosts = "abcdefghijklmnopqrstuvwxyz"

        input_counts = []
        expected_count = 0
        for count, host in enumerate(hosts, 1):
            expected_count += count
            data = {'hostname': host, 'key': 'totals', 'count': count}
            input_counts.append(AnalyticOutput.from_data([data]))

        random.shuffle(input_counts)

        reducer = get_reducer('| count')
        output_results = reducer(input_counts)

        self.assertEqual(len(output_results), 1)
        result = output_results[0]

        self.assertEqual(len(result.events), 1)
        data = result.events[0].data

        self.assertEqual(data['hosts'], list(hosts))
        self.assertEqual(data['total_hosts'], len(hosts))
        self.assertEqual(data['count'], expected_count)

    def test_aggregate_single_key_counts(self):
        """Test that counts are aggregated correctly with a single key."""
        input_results = [
            ('host1', 'key1', 2),
            ('host2', 'key1', 4),
            ('host3', 'key3', 2),
            ('host4', 'key1', 7),
            ('host5', 'key1', 9),
            ('host2', 'key2', 5),
            ('host1', 'key4', 3),
            ('host6', 'key3', 1),
        ]

        random.shuffle(input_results)
        input_counts = [{'hostname': h, 'key': k, 'count': c} for h, k, c in input_results]

        expected_counts = [
            ('key3', ['host3', 'host6'], 2 + 1),
            ('key4', ['host1'], 3),
            ('key2', ['host2'], 5),
            ('key1', ['host1', 'host2', 'host4', 'host5'], 2 + 4 + 7 + 9),
        ]

        reducer = get_reducer('| count a', config={'flatten': True})
        output_results = reducer(input_counts)

        self.assertEqual(len(expected_counts), len(output_results))
        for (key, hosts, count), event in zip(expected_counts, output_results):
            data = event.data  # type: dict
            self.assertEqual(key, data['key'])
            self.assertEqual(hosts, data['hosts'])
            self.assertEqual(len(hosts), data['total_hosts'])
            self.assertEqual(count, data['count'])

    def test_aggregate_multiple_key_counts(self):
        """Test that counts are aggregated correctly with multiple keys."""
        input_results = [
            ('host1', ['key1', 'key2', 'key3'], 2),
            ('host2', ['key1', 'key2', 'key3'], 4),
            ('host3', ['key1', 'key2', 'key3'], 2),
            ('host4', ['key1', 'key2', 'key5'], 7),
            ('host5', ['key1', 'key2', 'key5'], 9),
            ('host2', ['key2', 'key3', 'key4'], 5),
            ('host1', ['key4', 'key2', 'key5'], 3),
        ]

        random.shuffle(input_results)
        input_counts = [Event.from_data({'hostname': h, 'key': k, 'count': c}) for h, k, c in input_results]

        expected_counts = [
            (('key4', 'key2', 'key5'), ['host1'], 3),
            (('key2', 'key3', 'key4'), ['host2'], 5),
            (('key1', 'key2', 'key3'), ['host1', 'host2', 'host3'], 2 + 4 + 2),
            (('key1', 'key2', 'key5'), ['host4', 'host5'], 7 + 9),
        ]

        reducer = get_reducer('| count a b c', config={'flatten': True})
        reduced_counts = reducer(input_counts)

        self.assertEqual(len(expected_counts), len(reduced_counts))
        for (key, hosts, count), event in zip(expected_counts, reduced_counts):
            data = event.data  # type: dict
            self.assertEqual(key, data['key'])
            self.assertEqual(hosts, data['hosts'])
            self.assertEqual(len(hosts), data['total_hosts'])
            self.assertEqual(count, data['count'])

    def test_map_reduce_analytics(self):
        """Test map reduce functionality of python engines."""
        input_events = defaultdict(list)
        host_results = []

        for i, host in enumerate("abcdefghijklmnop"):
            events = []
            for event_number in range(10):
                data = {'number': event_number, 'a':  host + '-a-' + str(event_number), 'b': -event_number}
                events.append(Event.from_data(data))
            input_events[host] = events

        query_text = 'generic where true | sort a | head 5 | sort b'
        analytic = parse_analytic({'query': query_text, 'metadata': {'id': 'test-analytic'}})
        host_engine = get_engine(analytic)

        # Map across multiple 'hosts'
        for hostname, host_events in input_events.items():
            for result in host_engine(host_events):  # type: AnalyticOutput
                for event in result.events:
                    event.data['hostname'] = hostname
                host_results.append(result)

        # Reduce across multiple 'hosts'
        reducer = get_reducer(analytic)
        reduced_results = reducer(host_results)

        expected_a = ['a-a-{}'.format(value) for value in range(10)][:5][::-1]
        actual_a = [event.data['a'] for result in reduced_results for event in result.events]
        self.validate_results(actual_a, expected_a, query_text)

    def test_post_processor(self):
        """Test that post-processing of analytic results works."""
        data = [Event.from_data({'num': i}) for i in range(100)]
        query = '| head 10'
        processor = get_post_processor(query, {'flatten': True})
        results = processor(data)
        self.validate_results(results, data[:10], query)

    def test_special_pipes(self):
        """Make sure that the extra pipes are working as intended."""
        query = 'process where true | unique opcode | count'
        config = {'flatten': True}
        results = self.get_output(queries=[parse_query(query)], config=config)
        self.assertEqual(results[0].data['count'], 3, "Expected 3 unique process opcodes")

        query = 'process where true | count opcode'
        results = self.get_output(queries=[parse_query(query)], config=config)
        opcodes = set(event.data['key'] for event in results)
        self.assertEqual(len(results), 3, "Expected 3 unique process opcodes in the data set")
        self.assertEqual(opcodes, set([1, 2, 3]), "Some opcodes were missing")

        query = 'process where true | unique unique_pid | count opcode'
        results = self.get_output(queries=[parse_query(query)], config=config)
        opcodes = [event.data['key'] for event in results]
        self.assertEqual(len(results), 2, "Expected 2 opcodes")
        self.assertEqual(opcodes, [1, 3], "Received or missing opcodes")

        query = 'process where true | filter process_name == "svchost.exe"'
        results = self.get_output(queries=[parse_query(query)], config=config)
        self.assertGreater(len(results), 1, "Filter pipe failed")
        for event in results:
            self.assertEqual(event.data['process_name'].lower(), "svchost.exe")

        query = 'process where length(md5) > 0 | count md5 command_line'
        results = self.get_output(queries=[parse_query(query)], config=config)
        self.assertGreater(len(results), 1, "Count pipe returned no results")
        sorted_results = list(sorted(results, key=lambda e: (e.data['count'], e.data['key'])))
        self.assertListEqual(sorted_results, results, "Count didn't output expected results")

    @staticmethod
    def _custom_echo(x):
        return x

    @staticmethod
    def _custom_reverse(x):
        return x[::-1]

    def test_custom_functions(self):
        """Custom functions in python."""
        config = {'flatten': True}
        query = "process where echo(process_name) == \"SvcHost.*\" and command_line == \"* -k *NetworkRes*d\""
        output = self.get_output(queries=[parse_query(query)], config=config)
        event_ids = [event.data['serial_event_id'] for event in output]
        self.validate_results(event_ids, [15, 16, 25], "Custom function 'echo' failed")

        query = "process where length(user_domain)>0 and reverse(echo(user_domain)) = \"YTIROHTUA TN\" | tail 3"
        output = self.get_output(queries=[parse_query(query)], config=config)
        event_ids = [event.data['serial_event_id'] for event in output]
        self.validate_results(event_ids, [43, 45, 52], "Custom function 'reverse'")

    def test_analytic_output(self):
        """Confirm that analytics return the same results as queries."""
        analytics = [q['analytic'] for q in self.get_example_queries()]

        # Make sure they also work when run as analytics
        analytic_engine = PythonEngine({'data_source': 'endgame'})
        analytic_engine.add_custom_function('echo', self._custom_echo)
        analytic_engine.add_custom_function('reverse', self._custom_reverse)
        analytic_engine.add_analytics(analytics)
        output_ids = defaultdict(list)

        def add_analytic_output(output):  # type: (AnalyticOutput) -> None
            for event in output.events:
                output_ids[output.analytic_id].append(event.data['serial_event_id'])

        analytic_engine.add_output_hook(add_analytic_output)
        analytic_engine.stream_events(self.get_events())
        for analytic in analytics:
            query = analytic.name
            expected_ids = analytic.metadata['_info']['expected_event_ids']
            actual_ids = output_ids[analytic.id]
            self.validate_results(actual_ids, expected_ids, query)
