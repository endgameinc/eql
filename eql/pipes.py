"""EQL Pipes."""
from .ast import PipeCommand
from .schema import Schema, EVENT_TYPE_GENERIC
from .types import TypeHint, NodeInfo  # noqa: F401

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
    additional_types = tuple(prim.require_dynamic() for prim in TypeHint.primitives())
    minimum_args = 1


@PipeCommand.register('count')
class CountPipe(ByPipe):
    """Counts number of events that match a field, or total number of events if none specified."""

    minimum_args = 0

    @classmethod
    def output_schemas(cls, arguments, event_schemas):
        # type: (list[NodeInfo], list[Schema]) -> list[Schema]
        """Generate the output schema and determine the ``key`` field dynamically."""
        if len(arguments) == 0:
            key_hint = TypeHint.String.value
        elif len(arguments) == 1:
            key_hint = arguments[0].type_info.value
        else:
            key_hint = [arg.type_info.value for arg in arguments]

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

    argument_types = [TypeHint.Numeric.require_literal()]
    minimum_args = 0
    DEFAULT = 50

    @classmethod
    def validate(cls, arguments):
        """After performing type checks, validate that the count is greater than zero."""
        index = super(HeadPipe, cls).validate(arguments)
        if index is None and cls([arg.node for arg in arguments]).count <= 0:
            index = 0
        return index

    @property
    def count(self):  # type: () -> int
        """Get the number of elements to emit."""
        if len(self.arguments) == 0:
            return self.DEFAULT
        return self.arguments[0].value


@PipeCommand.register('tail')
class TailPipe(PipeCommand):
    """Node representing the tail pipe, analogous to the unix tail command."""

    argument_types = [TypeHint.Numeric.require_literal()]
    minimum_args = 0
    DEFAULT = 50

    @classmethod
    def validate(cls, arguments):
        """After performing type checks, validate that the count is greater than zero."""
        index = super(TailPipe, cls).validate(arguments)
        if index is None and cls([arg.node for arg in arguments]).count <= 0:
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
    def output_schemas(cls, arguments, event_schemas):
        # type: (list, list[Schema]) -> list[Schema]
        """Generate the output schema and determine the ``key`` field dyanmically."""
        if len(event_schemas) < 1:
            return event_schemas

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

    argument_types = [TypeHint.primitives()]

    @property
    def expression(self):
        """Get the filter expression."""
        return self.arguments[0]
