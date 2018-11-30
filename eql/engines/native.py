"""EQL engine in native python."""
from __future__ import print_function

import json
import re
from collections import defaultdict, deque, OrderedDict, namedtuple

from eql.ast import *  # noqa
from eql.engines.base import BaseEngine, BaseTranspiler, NodeMethods, Event, AnalyticOutput
from eql.schema import EVENT_TYPE_ANY, EVENT_TYPE_GENERIC
from eql.utils import is_string, is_number, get_type_converter, to_unicode

PIPE_EOF = object()


class Scope(namedtuple('Scope', ['events', 'variables'])):
    """Used for passing variables that may be referenced by nested callback functions."""

    @property
    def event(self):
        """Get the first event."""
        return self.events[0]

    def call(self, fn, *args):
        """Call a function by temporarily adding variables to the scope."""
        size = len(self.variables)
        self.variables.extend(args)
        status = fn(self)
        self.variables[size:] = []
        return status


class PythonEngine(BaseEngine, BaseTranspiler):
    """Converter from EQL to Python callbacks."""

    converters = NodeMethods()
    pipes = NodeMethods()
    reducers = NodeMethods()
    special_functions = NodeMethods()

    def __init__(self, config=None):
        """Create a python engine for EQL."""
        super(PythonEngine, self).__init__(config)
        self._output_hooks = []
        self._any_event_hooks = []
        # Any new list of hooks will automatically inherit from the global list
        self._event_hooks = defaultdict(lambda: list(self._any_event_hooks))
        self._functions = {}
        self._query_multiple_events = True
        self._in_pipe = False
        self._query_pipes = []
        self._reducer_hooks = defaultdict(list)
        self.host_key = self.get_config('host_key', 'hostname')

        if self.get_config('data_source') == 'endgame':
            self.process_subtype = "opcode"
            self.create_values = (1, 3)
            self.terminate_values = (2, 4)
        else:
            self.process_subtype = "subtype"
            self.create_values = ["create"]
            self.terminate_values = ["terminate"]

        self.add_custom_function('length', self._length)
        self.add_custom_function('arrayContains', self._array_contains)
        self.add_custom_function('safe', self._convert_safe_callback)

        # String functions
        self.add_custom_function('match', self._match)
        self.add_custom_function('matchLite', self._match)
        self.add_custom_function('startsWith', self._str_starts_with)
        self.add_custom_function('endsWith', self._str_ends_width)
        self.add_custom_function('stringContains', self._str_contains)
        self.add_custom_function('indexOf', self._str_index_of)
        self.add_custom_function('substring', self._str_substring)
        self.add_custom_function('string', to_unicode)
        self.add_custom_function('concat', self._concat)
        self.add_custom_function('number', self._number)

        # Math functions
        self.add_custom_function('add', self._add)
        self.add_custom_function('subtract', self._subtract)
        self.add_custom_function('multiply', self._multiply)
        self.add_custom_function('divide', self._divide)
        self.add_custom_function('modulo', self._modulo)

        self._scoped = list()

        for name, fn in self.get_config('functions', {}).items():
            self.add_custom_function(name, fn)

        self._output_hooks.extend(self.get_config('hooks', []))
        self.flatten = self.get_config('flatten', False)

        if self.get_config('print', False):
            self._default_emitter = self.print_events
        elif self.flatten:
            self._default_emitter = self.output_single_events
        else:
            self._default_emitter = self.get_result_emitter()

    @staticmethod
    def _length(value):
        if value is None:
            return 0
        else:
            return len(value)

    @staticmethod
    def _match(pattern, value):
        return value is not None and re.match(pattern, value, re.IGNORECASE) is not None

    @staticmethod
    def _str_starts_with(a, b):  # type: (str, str) -> bool
        return is_string(a) and is_string(b) and a.lower().startswith(b.lower())

    @staticmethod
    def _str_ends_width(a, b):  # type: (str, str) -> bool
        return is_string(a) and is_string(b) and a.lower().endswith(b.lower())

    @staticmethod
    def _str_contains(a, b):  # type: (str, str) -> bool
        return is_string(a) and is_string(b) and b.lower() in a.lower()

    @staticmethod
    def _str_index_of(a, b, start=0):  # type: (str, str, int) -> int
        if is_string(a) and is_string(b):
            a = a.lower()
            b = b.lower()
            if b in a[start:]:
                return a.index(b, start)

    @staticmethod
    def _add(a, b):  # type: (int|float, int|float) -> (int|float)
        return (a or 0) + (b or 0)

    @staticmethod
    def _subtract(a, b):  # type: (int|float, int|float) -> (int|float)
        return (a or 0) - (b or 0)

    @staticmethod
    def _divide(a, b):  # type: (int|float, int|float) -> (int|float)
        if not b:
            return float('NaN')
        return (a or 0) / b

    @staticmethod
    def _multiply(a, b):  # type: (int|float, int|float) -> (int|float)
        return (a or 0) * (b or 0)

    @staticmethod
    def _modulo(a, b):  # type: (int|float, int|float) -> (int|float)
        if not b:
            return float('NaN')
        return (a or 0) % b

    @staticmethod
    def _str_substring(a, start=None, end=None):  # type: (str, int, int) -> str
        if is_string(a):
            return a[start:end]

    @staticmethod
    def _concat(*args):
        return u"".join(to_unicode(arg) for arg in args)

    @staticmethod
    def _number(arg, base=10):  # type: (str, int) -> int|float
        if is_number(arg):
            return arg
        elif is_string(arg):
            if '.' in arg:
                return float(arg)
            if arg.startswith('0x'):
                arg = arg[2:]
                base = 16
            try:
                return int(arg, base)
            except ValueError:
                return None

    @staticmethod
    def _array_contains(array, value):
        if array is None:
            return False

        if is_string(value):
            value = value.lower()

        for item in array:
            if item == value:
                return True
            elif is_string(item) and item.lower() == value:
                return True
        return False

    def print_event(self, event):  # type: (Event) -> None
        """Print an event to stdout."""
        print(json.dumps(event.data, sort_keys=True))

    def print_events(self, events):
        """Print an array of events to stdout."""
        if events is not PIPE_EOF:
            for event in events:
                self.print_event(event)

    def _to_hooks(self, item):
        for hook in self._output_hooks:
            hook(item)

    def output_single_events(self, events):
        """Output a list of events to all callbacks."""
        if events is not PIPE_EOF:
            for event in events:
                self._to_hooks(event)

    def get_result_emitter(self, analytic_id=None, next_pipe=None):
        """Get a function that returns results for an analytic."""
        if next_pipe is None:
            next_pipe = self._to_hooks

        def flatten_with_id(events):
            if events is not PIPE_EOF:
                events = [event.copy() for event in events]
                for event in events:
                    event.data['analytic_id'] = analytic_id
                self.output_single_events(events)

        def output_results(events):  # type: (list[Event]) -> None
            if events is not PIPE_EOF:
                result = AnalyticOutput(analytic_id, events)
                next_pipe(result)

        if self.flatten:
            if not analytic_id:
                return self.output_single_events
            else:
                return flatten_with_id
        else:
            return output_results

    def convert(self, node, piped=False, scoped=False):
        """Convert an eql AST to a python callback function.

        :param EqlNode node: The eql AST
        :param bool piped: Create a pipe callback for multiple events
        :param bool scoped: Wrap the callback with variable scoping
        :return A python callback function that takes an event.
        :rtype: (Event|Scope|list[Event]) -> object
        """
        cb = PythonEngine.converters(self, node)
        if not scoped:
            return cb
        elif piped:
            def wrapped(events):
                return cb(Scope(events, []))

            return wrapped
        else:
            def wrapped(event):
                return cb(Scope([event], []))

            return wrapped

    def _convert_key(self, args, scoped=True, piped=False):
        """Convert a tuple of AST nodes to a callback function that returns a key.

        :param list[Event] args:
        :param bool piped: Create a pipe callback for multiple events
        :param bool scoped: Wrap the callback with variable scoping
        :rtype: (Scope|Event|list[Event]) -> tuple[object]
        """
        if len(args) == 0:
            return lambda e: None

        elif len(args) == 1:
            return self.convert(args[0], scoped=scoped, piped=piped)

        callbacks = [self.convert(arg, scoped=scoped, piped=piped) for arg in args]

        def to_tuple_callback(value):
            return tuple(callback(value) for callback in callbacks)

        return to_tuple_callback

    def _convert_tuple(self, args):
        """Convert a tuple of AST nodes to a callback function that returns a tuple of values.

        :param list[EqlNode] args:
        :rtype: (eql.engines.base.Event) -> tuple[object]
        """
        callbacks = [self.convert(arg) for arg in args]

        if len(callbacks) == 0:
            tup = tuple()
            return lambda e: tup

        def to_tuple_callback(value):
            return tuple(callback(value) for callback in callbacks)

        return to_tuple_callback

    def convert_pipe(self, node, next_pipe):
        """Convert an EQL pipe into a callback function.

        :param PipeCommand node: The original EQL pipe
        :param (list[eql.engines.base.Event]) -> None next_pipe: An already converted pipe
        :rtype: (list[eql.engines.base.Event]) -> None
        """
        return self.pipes(self, node, next_pipe)

    def convert_reducer(self, node, next_pipe):
        """Convert an EQL reducer into a callback function.

        :param PipeCommand node: The original EQL pipe
        :param (list[eql.engines.base.Event]) -> None next_pipe: An already converted reducer
        :rtype: (list[eql.engines.base.Event]) -> None
        """
        return self.reducers(self, node, next_pipe)

    @converters.add(Not)
    def _negate(self, node):  # type: (Not) -> callable
        get_value = self.convert(node.term)

        def negate(scope):  # type: (Scope) -> bool
            return not get_value(scope)

        return negate

    @converters.add(Literal)
    @converters.add(String)
    @converters.add(Boolean)
    @converters.add(Null)
    @converters.add(Number)
    def _get_value(self, node):  # type: (Literal) -> callable
        literal_value = node.value
        return lambda scope: literal_value

    @converters.add(Field)
    def _get_field(self, node):  # type: (Field) -> callable
        def walk_path(value):
            for key in node.path:
                if value is None:
                    break
                elif isinstance(value, dict):
                    value = value.get(key)
                elif key < len(value):
                    value = value[key]
                else:
                    return

            return value

        # if the variable is a dynamic variable, then find it from the list of named arguments
        if node.base in self._scoped:
            for pos, name in enumerate(self._scoped):
                if name == node.base:
                    index = pos

            def dynamic_func(scope):  # type: (Scope) -> object
                return walk_path(scope.variables[index])

            return dynamic_func

        if self._in_pipe:
            # Check if this is querying for any events
            if self._query_multiple_events:
                index, node = node.query_multiple_events()
            else:
                index = 0

            def pipe_callback(scope):  # type: (Scope) -> object
                event = scope.events[index]
                return walk_path(event.data.get(node.base))

            return pipe_callback
        else:
            def query_event_callback(scope):  # type: (Scope) -> object
                return walk_path(scope.event.data.get(node.base))

            return query_event_callback

    @staticmethod
    def _is_name(node):  # type: (EqlNode) -> bool
        return isinstance(node, Field) and not node.path

    def _create_custom_callback(self, arguments, body):
        """Convert an EQL callback with named arguments.

        :param list[Field] arguments: List of named arguments for the callback function
        :param Expression arguments: List of named arguments for the callback function
        :rtype: (Scope, object) -> object
        """
        names = [arg.base for arg in arguments]
        size = len(self._scoped)
        self._scoped.extend(names)

        callback = self.convert(body)
        self._scoped[size:] = []

        return callback

    @special_functions.add('safe')
    def _convert_safe_callback(self, arguments):
        get_value = self.convert(arguments[0])

        def callback(scope):
            try:
                return get_value(scope)
            except:
                pass

        return callback

    @special_functions.add('wildcard')
    def _convert_wildcard(self, arguments):
        patterns = []
        for literal in arguments[1:]:
            regex = re.escape(literal.value.lower())
            regex = "^" + regex.replace('\\*', '.*?') + "$"
            patterns.append(regex)

        compound = re.compile('|'.join(patterns), re.I)
        get_source = self.convert(arguments[0])

        def check_match(scope):
            text = get_source(scope)
            return text is not None and compound.match(text) is not None

        return check_match

    @special_functions.add('arraySearch')
    def _convert_array_search(self, arguments):
        node = FunctionCall('arraySearch', arguments)
        if len(arguments) == 3 and self._is_name(arguments[1]):
            array, name, body = arguments
            get_array = self.convert(array)
            callback = self._create_custom_callback([name], body)

            def walk_array(scope):  # type: (Scope) -> bool
                array = get_array(scope)
                if isinstance(array, list):
                    for item in array:
                        if scope.call(callback, item):
                            return True
                return False

            return walk_array
        raise TypeError(u"Invalid signature {}".format(node))

    @converters.add(FunctionCall)
    def _get_function_call(self, node):  # type: (FunctionCall) -> callable
        name = node.name
        if name in self.special_functions:
            unbound = self.special_functions[node.name]
            return unbound(self, node.arguments)

        func = self._functions[node.name]
        get_arguments = self._convert_tuple(node.arguments)

        def wrapped_function(scope):  # type: (Scope) -> bool
            return func(*get_arguments(scope))

        return wrapped_function

    @converters.add(InSet)
    def _check_in_set(self, node):  # type: (InSet) -> callable
        if all(isinstance(item, Literal) for item in node.container):
            values = set()
            for item in node.container:
                value = item.value
                if is_string(value):
                    values.add(value.lower())
                else:
                    values.add(value)

            get_value = self.convert(node.expression)

            def callback(scope):  # type: (Scope) -> bool
                check_value = get_value(scope)
                if is_string(check_value):
                    check_value = check_value.lower()
                return check_value in values

            return callback

        else:
            return self.convert(node.synonym)

    @converters.add(Comparison)
    def _compare(self, node):  # type: (Comparison) -> callable
        get_left = self.convert(node.left)
        get_right = self.convert(node.right)

        def types_match(x, y):
            return (
                    type(x) == type(y) or
                    is_string(x) and is_string(y) or
                    is_number(x) and is_number(y)
            )

        def equals(x, y):
            if not types_match(x, y):
                return False
            elif is_string(x):
                return x.lower() == y.lower()
            else:
                return x == y

        # Create different comparison functions so that this if statement doesn't need to be evaluated every time
        if node.comparator == Comparison.EQ:
            compare = equals
        elif node.comparator == Comparison.NE:
            def compare(x, y):
                return not equals(x, y)
        elif node.comparator == Comparison.LT:
            def compare(x, y):
                return types_match(x, y) and x < y
        elif node.comparator == Comparison.LE:
            def compare(x, y):
                return types_match(x, y) and x <= y
        elif node.comparator == Comparison.GT:
            def compare(x, y):
                return types_match(x, y) and x > y
        elif node.comparator == Comparison.GE:
            def compare(x, y):
                return types_match(x, y) and x >= y
        else:
            raise NotImplementedError("Unknown comparator {}".format(node.comparator))

        def callback(scope):  # type: (Scope) -> bool
            left = get_left(scope)
            right = get_right(scope)
            return compare(left, right)

        return callback

    @converters.add(And)
    def _convert_and(self, node):  # type: (CompoundTerm) -> callable
        get_terms = [self.convert(term) for term in node.terms]

        def and_terms(scope):  # type: (Scope) -> bool
            return all(get_term(scope) for get_term in get_terms)

        return and_terms

    @converters.add(Or)
    def _convert_or(self, node):  # type: (CompoundTerm) -> callable
        get_terms = [self.convert(term) for term in node.terms]

        def or_terms(scope):  # type: (Scope) -> bool
            return any(get_term(scope) for get_term in get_terms)

        return or_terms

    @pipes.add(CountPipe)
    def _convert_count_pipe(self, node, next_pipe):  # type: (CountPipe, callable) -> callable
        host_key = self.host_key
        if len(node.arguments) == 0:
            # Counting only the total
            summary = {'key': 'totals', 'count': 0}
            hosts = set()

            def count_total_callback(events):
                if events is PIPE_EOF:
                    if len(hosts):
                        summary['total_hosts'] = len(hosts)
                        summary['hosts'] = list(sorted(hosts))

                    next_pipe([Event(EVENT_TYPE_GENERIC, 0, summary)])
                    next_pipe(PIPE_EOF)
                else:
                    summary['count'] += 1
                    if host_key in events[0].data:
                        hosts.add(events[0].data[host_key])

            return count_total_callback

        else:
            get_key = self._convert_key(node.arguments, scoped=True, piped=True)
            count_table = defaultdict(lambda: {'count': 0, 'hosts': set()})

            def count_tuple_callback(events):  # type: (list[Event]) -> None
                if events is PIPE_EOF:
                    # This may seem a little tricky, but we need to effectively learn the type(s) to perform comparison
                    # Python 3 doesn't allow you to use a key function that returns various types
                    converter = get_type_converter(count_table)
                    converted_count_table = {converter(k): v for k, v in count_table.items()}
                    total = sum(tbl['count'] for tbl in count_table.values())

                    for key, details in sorted(converted_count_table.items(), key=lambda kv: (kv[1]['count'], kv[0])):
                        hosts = details.pop('hosts')
                        if len(hosts):
                            details['hosts'] = list(sorted(hosts))
                            details['total_hosts'] = len(hosts)

                        details['key'] = key
                        details['percent'] = float(details['count']) / total
                        next_pipe([Event(EVENT_TYPE_GENERIC, 0, details)])
                    next_pipe(PIPE_EOF)
                else:
                    key = get_key(events)

                    count_table[key]['count'] += 1
                    if host_key in events[0].data:
                        count_table[key]['hosts'].add(events[0].data[host_key])

            return count_tuple_callback

    @pipes.add(FilterPipe)
    @reducers.add(FilterPipe)
    def _convert_filter_pipe(self, node, next_pipe):  # type: (FilterPipe, callable) -> callable
        check_filter = self.convert(node.expression, piped=True, scoped=True)

        def filter_callback(events):  # type: (list[Event]) -> None
            if events is PIPE_EOF:
                next_pipe(PIPE_EOF)
            elif check_filter(events):
                next_pipe(events)

        return filter_callback

    @pipes.add(HeadPipe)
    @reducers.add(HeadPipe)
    def _convert_head_pipe(self, node, next_pipe):  # type: (HeadPipe, callable) -> callable
        totals = [0]  # has to be mutable because of python scoping
        max_count = node.count

        def head_callback(events):
            if totals[0] < max_count:
                if events is PIPE_EOF:
                    next_pipe(PIPE_EOF)
                else:
                    totals[0] += 1
                    next_pipe(events)
                    if totals[0] == max_count:
                        next_pipe(PIPE_EOF)

        return head_callback

    @pipes.add(TailPipe)
    @reducers.add(TailPipe)
    def _convert_tail_pipe(self, node, next_pipe):  # type: (TailPipe, callable) -> callable
        output_buffer = deque(maxlen=node.count)

        def tail_callback(events):
            if events is PIPE_EOF:
                for output in output_buffer:
                    next_pipe(output)
                next_pipe(PIPE_EOF)
            else:
                output_buffer.append(events)

        return tail_callback

    @pipes.add(SortPipe)
    @reducers.add(SortPipe)
    def _convert_sort_pipe(self, node, next_pipe):  # type: (SortPipe, callable) -> callable
        output_buffer = []
        sort_key = self._convert_key(node.arguments, scoped=True, piped=True)

        def sort_callback(events):
            if events is PIPE_EOF:
                converter = get_type_converter(sort_key(buf) for buf in output_buffer)

                def get_converted_key(buffer_events):
                    return converter(sort_key(buffer_events))

                output_buffer.sort(key=get_converted_key)
                for output in output_buffer:
                    next_pipe(output)
                next_pipe(PIPE_EOF)
            else:
                output_buffer.append(events)

        return sort_callback

    @pipes.add(UniquePipe)
    @reducers.add(UniquePipe)
    def _convert_unique_pipe(self, node, next_pipe):  # type: (UniquePipe, callable) -> callable
        seen = set()
        get_unique_key = self._convert_key(node.arguments, scoped=True, piped=True)

        def unique_callback(events):
            if events is PIPE_EOF:
                next_pipe(PIPE_EOF)
            else:
                key = get_unique_key(events)
                if key not in seen:
                    seen.add(key)
                    next_pipe(events)

        return unique_callback

    @pipes.add(UniqueCountPipe)
    @reducers.add(UniqueCountPipe)
    def _aggregate_unique_counts(self, node, next_pipe):  # type: (CountPipe) -> callable
        """Aggregate counts coming into the pipe."""
        host_key = self.host_key
        get_unique_key = self._convert_key(node.arguments, scoped=True, piped=True)
        results = OrderedDict()

        def count_unique_callback(events):  # type: (list[Event]) -> None
            if events is PIPE_EOF:
                # Calculate the total
                total = sum(result[0].data['count'] for result in results.values())

                for result in results.values():
                    hosts = result[0].data.pop('hosts')  # type: set
                    if len(hosts) > 0:
                        result[0].data['hosts'] = list(sorted(hosts))
                        result[0].data['total_hosts'] = len(hosts)

                    result[0].data['percent'] = float(result[0].data['count']) / total
                    next_pipe(result)
                next_pipe(PIPE_EOF)

            else:
                # Create a copy of these, because they can be modified
                events = [events[0].copy()] + events[1:]
                piece = events[0].data
                key = get_unique_key(events)
                hosts = piece.pop('hosts', [])
                host = piece.pop(host_key, None)
                count = piece.pop('count', 1)

                if key not in results:
                    results[key] = events
                    match = piece
                    match['hosts'] = set()
                    match['count'] = count
                else:
                    match = results[key][0].data
                    match['count'] += count

                if host:
                    match['hosts'].add(host)
                else:
                    match['hosts'].update(hosts)

        return count_unique_callback

    @reducers.add(CountPipe)
    def _aggregate_counts(self, node, next_pipe):  # type: (CountPipe) -> callable
        """Aggregate counts coming into the pipe."""
        host_key = self.host_key
        if len(node.arguments) == 0:
            # Counting only the total
            result = {'key': 'totals', 'count': 0, 'hosts': set()}

            def count_total_aggregates(events):  # type: (list[Event]) -> None
                if events is PIPE_EOF:
                    hosts = result.pop('hosts')  # type: set
                    if len(hosts) > 0:
                        result['hosts'] = list(sorted(hosts))
                        result['total_hosts'] = len(hosts)

                    next_pipe([Event(EVENT_TYPE_GENERIC, 0, result)])
                    next_pipe(PIPE_EOF)
                else:
                    piece = events[0].data
                    result['count'] += piece['count']

                    if host_key in piece:
                        result['hosts'].add(piece[host_key])
                    elif 'hosts' in piece:
                        results['hosts'].update(piece['hosts'])

            return count_total_aggregates

        else:
            results = defaultdict(lambda: {'count': 0, 'hosts': set()})

            def count_tuple_callback(events):  # type: (list[Event]) -> None
                if events is PIPE_EOF:
                    converter = get_type_converter(results)
                    converted_results = {converter(k): v for k, v in results.items()}

                    total = sum(result['count'] for result in converted_results.values())

                    for key, result in sorted(converted_results.items(), key=lambda kr: (kr[1]['count'], kr[0])):
                        hosts = result.pop('hosts')  # type: set
                        if len(hosts) > 0:
                            result['hosts'] = list(sorted(hosts))
                            result['total_hosts'] = len(hosts)
                        result['key'] = key
                        result['percent'] = float(result['count']) / total
                        next_pipe([Event(EVENT_TYPE_GENERIC, 0, result)])
                    next_pipe(PIPE_EOF)
                else:
                    piece = events[0].data
                    key = events[0].data['key']
                    key = tuple(key) if len(node.arguments) > 1 else key
                    results[key]['count'] += piece['count']
                    if host_key in piece:
                        results[key]['hosts'].add(piece[host_key])
                    elif 'hosts' in piece:
                        results[key]['hosts'].update(piece['hosts'])

            return count_tuple_callback

    @converters.add(NamedSubquery)
    def _get_named_of(self, node):  # type: (NamedSubquery) -> callable
        if node.query_type == NamedSubquery.DESCENDANT:
            return self._get_descendant_of(node.query)
        elif node.query_type == NamedSubquery.CHILD:
            return self._get_child_of(node.query)
        elif node.query_type == NamedSubquery.EVENT:
            return self._get_event_of(node.query)
        else:
            raise ValueError("Unknown query type {}".format(node.query_type))

    def _get_descendant_of(self, node):  # type: (EventQuery) -> callable
        sources = set()
        descendants = set()
        dead_processes = set()
        process_subtype = self.process_subtype
        creates = self.create_values
        terminates = self.terminate_values

        @self.event_callback("process")
        def update_descendants(event):  # type: (Event) -> None
            ppid = event.data.get('ppid')
            pid = event.data.get('pid')
            subtype = event.data.get(process_subtype)

            for pending_pid in dead_processes:
                if pending_pid in descendants:
                    descendants.remove(pending_pid)
                if pending_pid in sources:
                    sources.remove(pending_pid)

            dead_processes.clear()

            if subtype in creates and pid == 4 and event.data.get('process_name') == "System":
                # Reset all state on a sensor or machine boot up
                descendants.clear()
                sources.clear()

            if subtype in creates and (ppid in descendants or ppid in sources):
                # Check if the parent matches
                descendants.add(pid)
            elif subtype in terminates:
                dead_processes.add(pid)

        ancestor_match = self.convert(node.query, scoped=True)

        @self.event_callback(node.event_type)
        def check_ancestor(event):  # type: (Event) -> None
            pid = event.data.get('pid', 0)
            if pid != 0 and ancestor_match(event):
                sources.add(pid)

        def check_if_descendant(scope):  # type: (Scope) -> bool
            return scope.event.data.get('pid') in descendants

        return check_if_descendant

    def _get_child_of(self, node):  # type: (EventQuery) -> callable
        parents = set()
        children = set()
        dead_processes = set()
        process_subtype = self.process_subtype
        creates = self.create_values
        terminates = self.terminate_values

        @self.event_callback("process")
        def update_children(event):  # type: (Event) -> None
            ppid = event.data.get('ppid')
            pid = event.data.get('pid')
            subtype = event.data.get(process_subtype)

            for pending_pid in dead_processes:
                if pending_pid in children:
                    children.remove(pending_pid)
                if pending_pid in parents:
                    parents.remove(pending_pid)

            dead_processes.clear()

            if subtype in creates and pid == 4 and event.data.get('process_name') == "System":
                # Reset all state on a sensor or machine boot up
                children.clear()
                parents.clear()

            if subtype in creates and (ppid in parents):
                # Check if the parent matches
                children.add(pid)
            elif subtype in terminates:
                dead_processes.add(pid)

        process_match = self.convert(node.query, scoped=True)

        @self.event_callback(node.event_type)
        def match_processes(event):  # type: (Event) -> None
            pid = event.data.get('pid', 0)
            if pid != 0 and process_match(event):
                parents.add(pid)

        def check_if_child(scope):  # type: (Scope) -> None
            return scope.event.data.get('pid') in children

        return check_if_child

    def _get_event_of(self, node):  # type: (EventQuery) -> callable
        processes = set()
        dead_processes = set()
        process_subtype = self.process_subtype
        creates = self.create_values
        terminates = self.terminate_values

        @self.event_callback("process")
        def purge_on_terminate(event):  # type: (Event) -> None
            pid = event.data.get('pid')
            subtype = event.data.get(process_subtype)

            for pending_pid in dead_processes:
                if pending_pid in processes:
                    processes.remove(pending_pid)

            dead_processes.clear()

            if subtype in creates and pid == 4 and event.data.get('process_name') == "System":
                # Reset all state on a sensor or machine boot up
                processes.clear()

            elif subtype in terminates:
                dead_processes.add(pid)

        process_match = self.convert(node.query, scoped=True)

        @self.event_callback(node.event_type)
        def match_processes(event):  # type: (Event) -> None
            pid = event.data.get('pid', 0)
            if pid != 0 and process_match(event):
                processes.add(pid)

        def check_for_match(scope):  # type: (Scope) -> None
            return scope.event.data.get('pid') in processes

        return check_for_match

    @converters.add(EventQuery)
    def _convert_event_query(self, node):  # type: (EventQuery) -> callable
        check_match = self.convert(node.query, scoped=True)
        expected_type = node.event_type

        def match_event_callback(event):  # type: (Event) -> bool
            return expected_type == event.type and check_match(event)

        if expected_type == EVENT_TYPE_ANY:
            return check_match
        else:
            return match_event_callback

    @converters.add(Join)
    def _convert_join(self, node, next_pipe):  # type: (Join, callable) -> callable
        size = len(node.queries)
        lookup = defaultdict(lambda: [None] * size)  # type: dict[object, list[Event]]

        def convert_join_term(subquery, position):  # type: (SubqueryBy, int) -> callable
            check_event = self.convert(subquery.query)
            get_join_value = self._convert_key(subquery.join_values, scoped=True)

            @self.event_callback(subquery.query.event_type)
            def join_event_callback(event):  # type: (Event) -> None
                if check_event(event):
                    join_value = get_join_value(event)
                    if lookup[join_value][position] is None:
                        lookup[join_value][position] = event
                        if all(event is not None for event in lookup[join_value]):
                            next_pipe(lookup[join_value])
                            lookup.pop(join_value)

        if node.close:
            check_close_event = self.convert(node.close.query)
            close_join_value = self._convert_key(node.close.join_values, scoped=True)

            @self.event_callback(node.close.query.event_type)
            def close_join_callback(event):  # type: (Event) -> None
                if check_close_event(event):
                    join_value = close_join_value(event)
                    lookup.pop(join_value, None)

        for pos, query in enumerate(node.queries):
            convert_join_term(query, pos)

    def _convert_sequence_term(self, subquery, position, size, lookups, next_pipe=None):
        # type: (SubqueryBy, int, int, list[dict[object, list[Event]]], callable) -> callable
        check_event = self.convert(subquery.query)
        get_join_value = self._convert_key(subquery.join_values, scoped=True)
        last_position = size - 1
        fork = bool(subquery.params.kv.get('fork', Boolean(False)).value)

        if position == 0:
            @self.event_callback(subquery.query.event_type)
            def start_sequence_callback(event):  # type: (Event) -> None
                if check_event(event):
                    join_value = get_join_value(event)
                    sequence = [event]
                    lookups[1][join_value] = sequence

        elif position < last_position:
            next_position = position + 1

            @self.event_callback(subquery.query.event_type)
            def continue_sequence_callback(event):  # type: (Event) -> None
                if len(lookups[position]) and check_event(event):
                    join_value = get_join_value(event)
                    if join_value in lookups[position]:
                        if fork:
                            sequence = list(lookups[position].get(join_value))
                        else:
                            sequence = lookups[position].pop(join_value)
                        sequence.append(event)
                        lookups[next_position][join_value] = sequence

        else:
            @self.event_callback(subquery.query.event_type)
            def finish_sequence(event):  # type: (Event) -> None
                if len(lookups[position]) and check_event(event):
                    join_value = get_join_value(event)
                    if join_value in lookups[position]:
                        if fork:
                            sequence = list(lookups[position].get(join_value))
                        else:
                            sequence = lookups[position].pop(join_value)
                        sequence.append(event)
                        next_pipe(sequence)

    @converters.add(TimeRange)
    def _convert_range(self, node):
        return int(node.delta.total_seconds() * self._time_unit)

    @converters.add(Sequence)
    def _convert_sequence(self, node, next_pipe):  # type: (Sequence, callable) -> callable
        # Two lookups can help avoid unnecessary calls
        size = len(node.queries)
        lookups = [{} for _ in range(size)]  # type: list[dict[object, list[Event]]]

        if 'maxspan' in node.params.kv:
            max_span = self.convert(node.params.kv['maxspan'])
            event_types = set(q.query.event_type for q in node.queries)

            @self.event_callback(*event_types)
            def check_timeout(event):  # type: (Event) -> None
                minimum_start = event.time - max_span
                for sub_lookup in lookups:
                    for join_key, sequence in list(sub_lookup.items()):
                        if sequence[0].time < minimum_start:
                            sub_lookup.pop(join_key)

        if node.close:
            check_close_event = self.convert(node.close.query)
            get_close_join_value = self._convert_key(node.close.join_values, scoped=True)

            @self.event_callback(node.close.query.event_type)
            def close_sequences(event):  # type: (Event) -> None
                if check_close_event(event):
                    join_value = get_close_join_value(event)
                    for sub_lookup in lookups:
                        if join_value in sub_lookup:
                            sub_lookup.pop(join_value)

        for pos, query in reversed(list(enumerate(node.queries))):
            # Create these in reverse order, so one event can't hit multiple callbacks to be propagated
            self._convert_sequence_term(query, pos, len(node.queries), lookups, next_pipe)

    def _get_pipe_chain(self, pipes, output_pipe=None, query_multiple=True):
        # type: (list[PipeCommand], callable) -> callable
        """Get a chain of pipes."""
        prev_query_value = self._query_multiple_events
        self._query_multiple_events = query_multiple
        output_pipe = output_pipe or self._default_emitter
        self._in_pipe = True

        for pipe in reversed(pipes):
            output_pipe = self.convert_pipe(pipe, output_pipe)

        self._in_pipe = False
        self._query_multiple_events = prev_query_value
        return output_pipe

    def _get_pipe_reducers(self, pipes, output_pipe=None, query_multiple=True):
        # type: (list[PipeCommand], callable, bool) -> callable
        """Get a chain of pipes."""
        prev_query_value = self._query_multiple_events
        self._query_multiple_events = query_multiple
        output_pipe = output_pipe or self._default_emitter
        self._in_pipe = True

        for pipe in reversed(pipes):
            output_pipe = self.convert_reducer(pipe, output_pipe)
            if isinstance(pipe, (CountPipe, UniqueCountPipe)):
                break
        else:
            # Sort these events by time
            next_pipe = output_pipe
            results = []

            def sort_results(events):  # type: (list[Event]) -> None
                if events is not PIPE_EOF:
                    results.append(events)
                else:
                    results.sort(key=lambda result: (max(event.time for event in result),
                                                     max(event.data.get('serial_event_id') for event in result)))
                    for result in results:
                        next_pipe(result)
                    next_pipe(PIPE_EOF)

            output_pipe = sort_results

        self._in_pipe = False
        self._query_multiple_events = prev_query_value
        return output_pipe

    @converters.add(PipedQuery)
    def _convert_piped_query(self, node, output_pipe=None):  # type: (PipedQuery, callable) -> callable
        base_query = node.first

        query_multiple = not isinstance(base_query, EventQuery)
        output_pipe = self._get_pipe_chain(node.pipes, output_pipe=output_pipe, query_multiple=query_multiple)
        self.register_output_pipe(output_pipe)

        if isinstance(base_query, EventQuery):
            event_query = base_query
            check_match = self._convert_event_query(event_query)

            @self.event_callback(event_query.event_type)
            def callback(event):  # type: (Event) -> None
                if check_match(event):
                    output_pipe([event])

        elif isinstance(base_query, Join):
            self._convert_join(base_query, output_pipe)

        elif isinstance(base_query, Sequence):
            self._convert_sequence(base_query, output_pipe)

        else:
            raise NotImplementedError("Unsupported {}".format(type(base_query).__name__))

    def _convert_analytic(self, analytic):  # type: (EqlAnalytic) -> callable
        analytic_id = analytic.id or analytic.name
        self._convert_piped_query(analytic.query, self.get_result_emitter(analytic_id))

    def add_custom_function(self, name, func):  # type: (str, function) -> None
        """Load a python function into the EQL engine."""
        self._functions[name] = func

    def add_analytic(self, analytic):  # type: (EqlAnalytic) -> None
        """Convert an analytic and load into the engine."""
        expanded_analytic = self.preprocessor.expand(analytic)
        self._convert_analytic(expanded_analytic)

    def add_query(self, query):  # type: (PipedQuery | EqlAnalytic) -> None
        """Convert an analytic and load into the engine."""
        query = self.preprocessor.expand(query)
        self._convert_piped_query(query)

    def add_queries(self, queries):
        """Add multiple queries to the engine."""
        for query in queries:
            self.add_query(query)

    def add_post_processor(self, query, analytic_id=None, output_pipe=None, query_multiple=False):
        # type: (PipedQuery, str, callable, bool) -> None
        """Register a query post-processor to perform additional filtering of results."""
        chain = self._get_pipe_chain(query.pipes, output_pipe, query_multiple=query_multiple)
        self._reducer_hooks[analytic_id].append(chain)

    def add_reducer(self, query, analytic_id=None, output_pipe=None):
        """Reduce the output from multiple queries.

        :param PipedQuery|EqlAnalytic query: The analytic to extra the reuce logic from
        :param str analytic_id: Optional analytic_id to add to AnalyticOutput results
        :param callable output_pipe: Next pipe to reduce to
        """
        if isinstance(query, EqlAnalytic):
            analytic_id = query.id or analytic_id
            output_pipe = self.get_result_emitter(query.id, output_pipe)
            query = query.query

        query_multiple = not isinstance(query.first, EventQuery)
        reduce_pipe_chain = self._get_pipe_reducers(query.pipes, output_pipe, query_multiple=query_multiple)

        # At this point output_pipe is the entry point to the reducer
        self._reducer_hooks[analytic_id].append(reduce_pipe_chain)

    def stream_event(self, event):  # type: (Event) -> None
        """Stream a single :class:`~Event` through the engine."""
        for hook in self._event_hooks[event.type]:
            hook(event)

    def finalize(self):
        """Send the engine an EOF signal, so that aggregating pipes can finish."""
        for pipe in self._query_pipes:
            pipe(PIPE_EOF)

        for analytic_id, reducers in self._reducer_hooks.items():
            for reducer in reducers:
                reducer(PIPE_EOF)

    def stream_events(self, events, finalize=True):
        """Stream :class:`~Event` objects through the engine."""
        for event in events:
            if not isinstance(event, Event):
                event = Event.from_data(event)
            self.stream_event(event)
        if finalize:
            self.finalize()

    def reduce_events(self, inputs, analytic_id=None, finalize=True):
        """Run an event through the reducers registered with :meth:`~add_reducer` and :meth:`~add_post_processor`.

        :param AnalyticOutput|Event|dict inputs: Mapped results to reduce
        :param str analytic_id: Optional analytic id to add to generated AnalyticOutput results
        :param bool finalize: Send the finalize signal when input is exhausted.
        """
        for data in inputs:
            if isinstance(data, AnalyticOutput):
                analytic_id = data.analytic_id or analytic_id
                events = data.events
            elif isinstance(data, Event):
                events = [data]
            elif isinstance(data, dict):
                events = [Event.from_data(data)]
            else:
                raise ValueError("Unable to reduce {}".format(data))

            for reducer in self._reducer_hooks[analytic_id]:
                reducer(events)

        if finalize:
            self.finalize()

    def add_event_callback(self, event_type, f):  # type: (int, callable) -> None
        """Register a callback for incoming events."""
        if event_type == EVENT_TYPE_ANY:
            # Note that if querying over all events, we need to preserve the order the hooks were created
            # So append them to all existing hook arrays
            self._any_event_hooks.append(f)
            for _, event_hooks in self._event_hooks.items():
                event_hooks.append(f)
        else:
            self._event_hooks[event_type].append(f)

    def event_callback(self, *event_types):
        """Get a decorator that registers a function as an event callback in the engine."""
        assert all(is_string(e) for e in event_types)

        def event_callback_decorator(f):
            for event_type in event_types:
                self.add_event_callback(event_type, f)
            return f

        return event_callback_decorator

    def register_output_pipe(self, f):
        """"Register a pipe, so that it can get called when the engine is closing."""
        self._query_pipes.append(f)

    def output_pipe(self, f):
        """"Decorator that registers a pipe, so that it can get called when the engine is closing."""
        self.register_output_pipe(f)
        return f

    def add_output_hook(self, f):
        """Register a callback to receive events as they are output from the engine."""
        self._output_hooks.append(f)


__all__ = (
    "PythonEngine",
)
