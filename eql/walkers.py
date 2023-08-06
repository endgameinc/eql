"""EQL walker classes."""
import re
from collections import defaultdict, deque
from contextlib import contextmanager

from .ast import TimeUnit
from .schema import Schema
from .types import NodeInfo, TypeHint
from .utils import is_string, to_unicode


__all__ = (
    "Walker",
    "RecursiveWalker",
    "ConfigurableWalker",
    "DepthFirstWalker",
)


DEFAULT_TIME_UNIT = 10000000  # Windows FileTime 0.1 microseconds


class Walker(object):
    """Base class that provides functionality for walking abstract syntax trees of eql.BaseNode."""

    __camelcache = {}

    def __init__(self):
        """Create the AST walker."""
        object.__init__(self)
        self._method_cache = defaultdict(dict)
        self.event_stack = []
        self.in_pipes = []
        self.base_event_types = []
        self.node_stack = []
        self.output_event_types = []

    def register_func(self, node_cls, func, prefix="_walk_"):
        """Register a callback function."""
        camelized = self.camelized(node_cls)
        method_name = prefix + camelized
        setattr(self, method_name, func)

    def iter_node(self, node):
        """Iterate through a syntax tree."""
        if isinstance(node, BaseNode):
            yield node

            for descendant in self.iter_node([v for v in node.iter_slots()]):
                yield descendant
        elif isinstance(node, (list, tuple)):
            for n in node:
                for descendant in self.iter_node(n):
                    yield descendant
        elif isinstance(node, dict):
            for n in self.iter_node(node.items()):
                yield n

    @classmethod
    def camelized(cls, node_cls):
        """Get the camelized name for the class."""
        if is_string(node_cls):
            class_name = node_cls
        else:
            if not isinstance(node_cls, type):
                node_cls = type(node_cls)
            class_name = node_cls.__name__
        if class_name not in cls.__camelcache:
            pass1 = re.sub('(.)([A-Z][a-z]+)', r'\1_\2', class_name)
            pass2 = re.sub('([a-z0-9])([A-Z])', r'\1_\2', pass1)
            cls.__camelcache[class_name] = to_unicode(pass2.lower())
        return cls.__camelcache[class_name]

    @property
    def current_event_type(self):
        """Get the active event type while walking."""
        if self.event_stack:
            return self.event_stack[-1]

    def _enter(self, node):
        self.event_stack.append(node.event_type)

    def _enter_event_query(self, node):
        self.event_stack.append(node.event_type)

    def _enter_piped_query(self, node):  # type: (PipedQuery) -> None
        self.output_event_types = []
        self.base_event_types = []
        if isinstance(node.first, EventQuery):
            self.base_event_types.append(node.first.event_type)
        else:
            self.base_event_types.extend(q.query.event_type for q in node.first.queries)

        self.output_event_types = self.base_event_types[:]

    def _enter_pipe_command(self, node):
        self.in_pipes = True

    def _enter_subquery_by(self, node):
        self.event_stack.append(node.query.event_type)

    def _exit_subquery_by(self, node):
        self.event_stack.pop()

    def _exit_event_query(self, node):
        self.event_stack.pop()

    def _exit_piped_query(self, node):
        self.base_event_types = []

    def _exit_pipe_command(self, node):
        """Update the output schemas as they change through each pipe."""
        self.in_pipes = False

        incoming_schema = [Schema({event_type: {}}) for event_type in self.output_event_types]
        output_schemas = node.output_schemas([NodeInfo(a, TypeHint.Unknown) for a in node.arguments], incoming_schema)
        self.output_event_types = [next(iter(s.schema.keys())) for s in output_schemas]

    def _walk_default(self, node, *args, **kwargs):
        return node

    def get_node_method(self, node_cls, prefix):  # type: (BaseNode, str) -> callable
        """Get the walk method for a node."""
        if not isinstance(node_cls, type):
            node_cls = type(node_cls)

        if node_cls in self._method_cache[prefix]:
            return self._method_cache[prefix][node_cls]

        queue = deque([node_cls])
        method = None

        while queue:
            next_cls = queue.popleft()
            method_name = prefix + self.camelized(next_cls)
            method = getattr(self, method_name, None)
            if callable(method):
                break

            queue.extend(next_cls.__bases__)

        method = method or getattr(self, prefix + "default", None)
        self._method_cache[prefix][node_cls] = method
        return method

    @property
    def active_node(self):
        """Get the active context."""
        return self.node_stack[-1]

    @property
    def parent_node(self):
        """Get the parent context."""
        return self.node_stack[-2]

    @contextmanager
    def set_context(self, node):
        """Push a node onto the context stack."""
        enter_method = self.get_node_method(node, prefix="_enter_")
        exit_method = self.get_node_method(node, prefix="_exit_")

        if callable(enter_method):
            enter_method(node)

        self.node_stack.append(node)

        try:
            yield node
        finally:
            self.node_stack.pop()
            if callable(exit_method):
                exit_method(node)

    def autowalk(self, node, *args, **kwargs):
        """Automatically walk built-in containers."""
        with self.set_context(node):
            if isinstance(node, list):
                return [self.walk(n, *args, **kwargs) for n in node]

            if isinstance(node, tuple):
                return tuple(self.walk(n, *args, **kwargs) for n in node)

            if isinstance(node, dict):
                return dict({self.walk(k, *args, **kwargs): self.walk(v, *args, **kwargs) for k, v in node.items()})

    def walk(self, node, *args, **kwargs):
        """Walk the syntax tree top-down."""
        rv = self.autowalk(node, *args, **kwargs)
        if rv is not None:
            return rv

        method = self.get_node_method(node, "_walk_")
        if callable(method):
            with self.set_context(node):
                return method(node, *args, **kwargs)


class RecursiveWalker(Walker):
    """Walker that will recursively walk and transform a tree."""

    def _walk_base_node(self, node, *args, **kwargs):  # type: (BaseNode) -> BaseNode
        cls = type(node)
        slots = self.walk([v for k, v in node.iter_slots()], *args, **kwargs)
        return cls(*slots)

    def copy_node(self, node):
        """Create a copy of a node."""
        return self.walk(node)


class DepthFirstWalker(Walker):
    """Walk an AST bottom up."""

    def walk(self, node, *args, **kwargs):
        """Walk the syntax tree top-down."""
        rv = self.autowalk(node, *args, **kwargs)
        if rv is not None:
            return rv

        method = self.get_node_method(node, "_walk_")

        if callable(method):
            with self.set_context(node):
                if isinstance(node, BaseNode):
                    slots = [self.walk(v, *args, **kwargs) for name, v in node.iter_slots()]
                    node = type(node)(*slots)
                return method(node, *args, **kwargs)

    def copy_node(self, node):
        """Create a copy of a node."""
        return RecursiveWalker().walk(node)


class ConfigurableWalker(RecursiveWalker):
    """Subclass for adding configurations to an walkers."""

    def __init__(self, config=None):
        """Create the walker with optional configuration."""
        self.config = config or {}
        self.stack = []
        self.time_unit = self.get_config('time_unit', DEFAULT_TIME_UNIT)  # type: int
        self._schema = None

        if self.get_config('schema', None) is not None:
            self._schema = Schema(**self.get_config('schema'))
        super(ConfigurableWalker, self).__init__()

    def convert_time_range(self, node):  # type: (eql.ast.TimeRange) -> (int|float)
        """Convert a time range to a timestamp delta."""
        tick_rate = TimeUnit.Seconds.as_milliseconds()
        node_ms = node.as_milliseconds() // tick_rate

        if not isinstance(self.time_unit, float) and self.time_unit > tick_rate and self.time_unit % tick_rate == 0:
            # strictly use integer math if we can safely divide the engine's rate by TimeUnits
            return self.time_unit * node_ms
        else:
            # if it doesn't evenly divide, resort to floating point math
            return float(self.time_unit) * node_ms

    @property
    def schema(self):
        """Get the current engine schema."""
        if self._schema is None:
            return Schema.current()
        return self._schema

    def get_config(self, name, default=None):
        """Get a property from the config dict."""
        return self.config.get(name, default)


# circular dependency
from .ast import BaseNode, EventQuery  # noqa: E402
