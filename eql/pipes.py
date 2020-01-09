"""EQL Pipes."""
from .ast import PipeCommand
from .schema import Schema, EVENT_TYPE_GENERIC, MIXED_TYPES
from .types import dynamic, NUMBER, literal, PRIMITIVES, EXPRESSION, get_type, BASE_STRING
from .utils import is_string

__all__ = (
    "list_pipes",
    "ByPipe",
    "HeadPipe",
    "TailPipe",
    "SortPipe",
    "UniquePipe",
    "CountPipe",
    "FilterPipe",
    "UniqueCountPipe",
)


def list_pipes():
    """"Get all of the current pipes."""
    return list(sorted(PipeCommand.lookup))


class ByPipe(PipeCommand):
    """Pipe that takes a value (field, function, etc.) as a key."""

    argument_types = []
    additional_types = dynamic(PRIMITIVES)
    minimum_args = 1


@PipeCommand.register('count')
class CountPipe(ByPipe):
    """Counts number of events that match a field, or total number of events if none specified."""

    minimum_args = 0

    @classmethod
    def output_schemas(cls, arguments, type_hints, event_schemas):
        # type: (list, list, list[Schema]) -> list[Schema]
        """Generate the output schema and determine the ``key`` field dyanmically."""
        if type_hints is None:
            type_hints = [MIXED_TYPES for _ in arguments]
        base_hints = [get_type(t) for t in type_hints]
        base_hints = [MIXED_TYPES if not is_string(t) else t for t in base_hints]
        if len(arguments) == 0:
            key_hint = BASE_STRING
        elif len(arguments) == 1:
            key_hint = base_hints[0]
        else:
            key_hint = base_hints

        return [Schema({
            EVENT_TYPE_GENERIC: {
                "count": "number",
                "percent": "number",
                "total_hosts": "number",
                "hosts": ["string"],
                "key":  key_hint,
            }
        }, allow_any=False, allow_generic=True)]


@PipeCommand.register('head')
class HeadPipe(PipeCommand):
    """Node representing the head pipe, analogous to the unix head command."""

    argument_types = [literal(NUMBER)]
    minimum_args = 0
    DEFAULT = 50

    @classmethod
    def validate(cls, arguments, type_hints=None):
        """After performing type checks, validate that the count is greater than zero."""
        index, arguments, type_hints = super(HeadPipe, cls).validate(arguments, type_hints)
        if index is None and cls(arguments).count <= 0:
            index = 0
        return index, arguments, type_hints

    @property
    def count(self):  # type: () -> int
        """Get the number of elements to emit."""
        if len(self.arguments) == 0:
            return self.DEFAULT
        return self.arguments[0].value


@PipeCommand.register('tail')
class TailPipe(PipeCommand):
    """Node representing the tail pipe, analogous to the unix tail command."""

    argument_types = [literal(NUMBER)]
    minimum_args = 0
    DEFAULT = 50

    @classmethod
    def validate(cls, arguments, type_hints=None):
        """After performing type checks, validate that the count is greater than zero."""
        index = super(TailPipe, cls).validate(arguments, type_hints)
        if index is None and cls(arguments).count <= 0:
            index = 0
        return index

    @property
    def count(self):  # type: () -> int
        """Get the number of elements to emit."""
        if len(self.arguments) == 0:
            return self.DEFAULT
        return self.arguments[0].value


@PipeCommand.register('sort')
class SortPipe(ByPipe):
    """Sorts the pipes by field comparisons."""


@PipeCommand.register('unique')
class UniquePipe(ByPipe):
    """Filters events on a per-field basis, and only outputs the first event seen for a field."""


@PipeCommand.register('unique_count')
class UniqueCountPipe(ByPipe):
    """Returns unique results but adds a count field."""

    minimum_args = 1

    @classmethod
    def output_schemas(cls, arguments, type_hints, event_schemas):
        # type: (list, list, list[Schema]) -> list[Schema]
        """Generate the output schema and determine the ``key`` field dyanmically."""
        event_schemas = list(event_schemas)
        first_event_type, = event_schemas[0].schema.keys()
        if any(v for v in event_schemas[0].schema.values()):
            event_schemas[0] = event_schemas[0].merge(Schema({
                first_event_type: {
                    "count": "number",
                    "total_hosts": "number",
                    "hosts": ["string"],
                    "percent": "number",
                }
            }, allow_any=False, allow_generic=True))
        return event_schemas


@PipeCommand.register('filter')
class FilterPipe(PipeCommand):
    """Takes data coming into an existing pipe and filters it further."""

    argument_types = [EXPRESSION]

    @property
    def expression(self):
        """Get the filter expression."""
        return self.arguments[0]
