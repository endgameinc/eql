"""Base class for constructing an analytic engine with analytics."""
from collections import namedtuple

from eql.ast import *  # noqa
from eql.parser import parse_definitions
from eql.schema import EVENT_TYPE_GENERIC, use_schema
from eql.utils import is_string


DEFAULT_TIME_UNIT = 10000000  # Windows FileTime 0.1 microseconds


class NodeMethods(dict):
    """Dictionary of methods to lookup by a key (usually a class)."""

    class UnknownNode(KeyError):
        """Unregistered node."""

    def add(self, key):
        """Add a callback method to the dictionary for a specific class.

        :param BaseNode key: The class of the object passed in
        """
        def decorator(f):
            """The function decorator that registers a method by node type."""
            assert key not in self
            self[key] = f
            return f

        return decorator

    def __call__(self, transpiler, node, *args, **kwargs):  # type: (BaseTranspiler, BaseNode) -> object
        """Call the bound method for a node."""
        cls = type(node)
        try:
            unbound = self[cls]
        except KeyError as e:
            raise NodeMethods.UnknownNode(e)
        return unbound(transpiler, node, *args, **kwargs)


class ConfigurableWalker(AstWalker):
    """Subclass for adding configurations to an walkers."""

    def __init__(self, config=None):
        """Create the walker with optional configuration."""
        self.config = config or {}
        self.stack = []
        self.schema = self.get_config('schema')
        super(ConfigurableWalker, self).__init__()

    def get_config(self, name, default=None):
        """Get a property from the config dict."""
        return self.config.get(name, default)


class BaseTranspiler(ConfigurableWalker):
    """Base Transpiler class for converting ASTs from one language to another."""

    converters = NodeMethods()
    renderers = NodeMethods()

    def __init__(self, config=None):
        """Instantiate the transpiler."""
        super(BaseTranspiler, self).__init__(config)
        self.config = config or {}
        self.stack = []  # type: list[BaseNode]
        self._time_unit = self.get_config('time_unit', DEFAULT_TIME_UNIT)  # type: int
        self._counter = 0

    def counter(self):
        """Increment counter and get current value."""
        self._counter += 1
        return self._counter

    def push(self, node):  # type: (BaseNode) -> None
        """Push node onto stack."""
        self.stack.append(node)

    def pop(self):  # type: () -> BaseNode
        """Pop node from stack."""
        return self.stack.pop()

    def pop_many(self, count):  # type: (int) -> list[BaseNode]
        """Pop multiple nodes from the stack."""
        popped = self.stack[-count:]
        self.stack[-count:] = []
        return popped

    def convert(self, node):  # type: (BaseNode) -> BaseNode
        """Convert an AST node with the registered converter functions."""
        return self.converters(self, node)


class BaseEngine(ConfigurableWalker):
    """Add and render EQL analytics to the generic engines."""

    def __init__(self, config=None):
        """Create the engine with an optional list of files."""
        super(BaseEngine, self).__init__(config)
        self.analytics = []  # type: list[EqlAnalytic]
        self.preprocessor = PreProcessor()

        with use_schema(self.schema):
            definitions = self.get_config('definitions', [])
            if is_string(definitions):
                definitions = parse_definitions(definitions)

            self.preprocessor.add_definitions(definitions)

            for path in self.get_config('definitions_files', []):
                with open(path, 'r') as f:
                    definitions = parse_definitions(f.read())
                self.preprocessor.add_definitions(definitions)

    def add_analytic(self, analytic):
        # type: (EqlAnalytic) -> None
        """Add a single analytic to the engine."""
        analytic = self.preprocessor.expand(analytic)
        self.analytics.append(analytic)

    def add_analytics(self, analytics):
        """Add multiple analytics to the engine."""
        for analytic in analytics:
            self.add_analytic(analytic)


# noinspection PyAbstractClass
class TextEngine(BaseEngine):
    """Converter for EQL to a target language script."""

    base_files = []
    transpiler_cls = BaseTranspiler
    extensions = {}

    def __init__(self, config=None):
        """Create the engine with an optional list of files."""
        super(TextEngine, self).__init__(config)
        self.files = []
        self.files.extend(self.base_files)
        self.converted = []
        self.transpiler = self.transpiler_cls(self.config)

    def _include_file(self, path):
        with open(path, 'r') as f:
            contents = '\n'.join(line.rstrip() for line in f)
            return contents

    def add_analytic(self, analytic):
        """Convert analytic and add to engine."""
        expanded_analytic = self.preprocessor.expand(analytic)
        converted = self.transpiler.convert(expanded_analytic)
        self.analytics.append(analytic)
        self.converted.append(converted)

    def include_files(self):
        """Generate lines of text for including files."""
        output_lines = []
        for path in self.files:
            output_lines.extend(self._include_file(path).splitlines())
        return output_lines

    def render_analytics(self):
        """Render the converted analytics as lines of text."""
        output_lines = []

        for converted_analytic in self.converted:
            rendered_analytic = converted_analytic.render()
            for line in rendered_analytic.splitlines():
                output_lines.append(line.rstrip())
        return output_lines

    def render(self, analytics_only=False):
        """Create the output text for the rendered engine and analytics."""
        output_lines = []

        if not analytics_only:
            output_lines.extend(self.include_files())

        output_lines.extend(self.render_analytics())
        return '\n'.join(output_lines)


class Event(namedtuple('Event', ['type', 'time', 'data'])):
    """Event for python engine in EQL."""

    @classmethod
    def from_data(cls, data):
        """Load an event from a dictionary.

        :param dict data: Dictionary with the event type, time, and keys.
        """
        data = data.get('data_buffer', data)
        timestamp = data.get('timestamp', 0)

        if is_string(data.get('event_type')):
            event_type = data['event_type']
        elif 'event_type_full' in data:
            event_type = data['event_type_full']
            if event_type.endswith('_event'):
                event_type = event_type[:-len('_event')]
        else:
            event_type = EVENT_TYPE_GENERIC

        return cls(event_type, timestamp, data)

    def copy(self):
        """Create a copy of the event."""
        data = self.data.copy()
        return Event(self.type, self.time, data)


def register_extension(ext):
    """Decorator used for registering TextEngines with specific file extensions building."""
    def decorator(cls):
        TextEngine.extensions[ext] = cls
        return cls
    return decorator


class AnalyticOutput(namedtuple('AnalyticOutput', ['analytic_id', 'events'])):
    """AnalyticOutput for python engine in EQL."""

    @classmethod
    def from_data(cls, events, analytic_id=None):  # type: (list[dict], str) -> AnalyticOutput
        """Load up an analytic output event."""
        return cls(analytic_id, [Event.from_data(e) for e in events])
