"""Eventing data schemas."""
import re
from .types import TypeHint
from .errors import EqlError
from .utils import is_string, is_number, ParserConfig

_global = None

EVENT_TYPE_ANY = 'any'
EVENT_TYPE_GENERIC = 'generic'

MIXED_TYPES = TypeHint.Unknown.value
IDENT_RE = re.compile(r"^[_a-zA-Z][a-zA-Z0-9_]*$")


is_primitive = set(b.value for b in TypeHint.primitives()).__contains__
primitive_lookup = {b.value: b for b in TypeHint.primitives()}


class Schema(ParserConfig):
    """Schema of all event types.

    Expected input format:
        {
          "process": {
            "process_name": "string",
            "command_line", "..."
          },
          "complex": {
            "flat": "string",
            "somearray": [],
            "somearray": ["string", "number", "boolean"],
            "field1": {"nested_field": "mixed", "doublenested": [{"sub1": "string", "sub2": "field"}]},
            "flexiblefield": {}
        }
    """

    _default_schema = None

    def __init__(self, events, allow_generic=True, allow_any=True, allow_missing=False):
        """Create a schema."""
        self.allow_generic = allow_generic
        self.allow_any = allow_any
        self.allow_missing = allow_missing
        self.schema = events

        if not self.validate_schema():
            raise EqlError("Invalid input schema {}".format(repr(events)))

        super(Schema, self).__init__(schema=self)

    def _validate_field_schema(self, field_schema):
        """Validate that a field schema is correct."""
        if is_string(field_schema) and len(field_schema) > 0:
            return is_primitive(field_schema) or field_schema == MIXED_TYPES
        elif isinstance(field_schema, (list, tuple)):
            return not field_schema or all(self._validate_field_schema(s) for s in field_schema)
        elif isinstance(field_schema, dict):
            for name, nested in field_schema.items():
                status = is_string(name) and IDENT_RE.match(name) and self._validate_field_schema(nested)
                if not status:
                    return False
            return True
        return False

    def validate_schema(self):
        """Validate that the schema is valid."""
        if not isinstance(self.schema, dict):
            return False

        for key, event_schema in self.schema.items():
            status = is_string(key) and isinstance(event_schema, dict)
            if not status:
                return False

            for name, field_schema in event_schema.items():
                status = is_string(name) and self._validate_field_schema(field_schema)
                if not status:
                    return False
        return True

    @classmethod
    def convert_to_type(cls, schema):
        """Convert a schema to the type system."""
        if schema == {} or schema == MIXED_TYPES:
            return TypeHint.Unknown, None
        elif isinstance(schema, dict):
            return TypeHint.Object, schema
        elif isinstance(schema, (list, tuple)):
            return TypeHint.Array, list(schema)
        else:
            return primitive_lookup[schema], None

    @classmethod
    def get_relative_path(cls, event_schema, path):
        """Validate a field against a schema."""
        base = path[0]
        subpath = path[1:]

        if is_number(base):
            # if the index is numeric, then the field must be an array
            if not isinstance(event_schema, (list, tuple)):
                return
            elif subpath:
                # if it's nested, then we have to enumerate over the union of nested schemas
                for subschema in event_schema:
                    hint = cls.get_relative_path(subschema, subpath)
                    if hint is not None:
                        return hint
                return
            else:
                return cls.convert_to_type(event_schema[0])
        elif isinstance(event_schema, (list, tuple)):
            # strings can't index into arrays
            return
        elif event_schema == {}:
            # if the event schema is wide open, then anything goes
            return cls.convert_to_type(event_schema)
        elif isinstance(event_schema, dict) and base in event_schema:
            if event_schema and subpath:
                # check if the current field is in the schema, and we still have to recurse
                return cls.get_relative_path(event_schema[base], subpath)
            else:
                # return the type hint if one exists
                return cls.convert_to_type(event_schema[base])

    def get_event_type_hint(self, event_type, path):
        """Validate that a field matches an event_type."""
        if not self.schema:
            return TypeHint.Unknown, None
        elif event_type == EVENT_TYPE_ANY:
            # search all of the known events, and find one that has this schema
            if self.allow_any:
                if not self.schema:
                    return TypeHint.Unknown, None
                for event_type in self.schema:
                    field_type = self.get_event_type_hint(event_type, path)
                    if field_type is not None:
                        return field_type
                if self.allow_missing:
                    return TypeHint.Unknown, None
        elif event_type in self.schema:
            # Convert the values to the expected string values or None
            type_hint = self.get_relative_path(self.schema[event_type], path)
            if type_hint is not None:
                return type_hint
            elif self.allow_missing:
                return TypeHint.Unknown, None
        elif event_type == EVENT_TYPE_GENERIC:
            if self.allow_generic:
                return TypeHint.Unknown, None

    def validate_event_type(self, event_type):
        """Validate that an event type is allowed by the schema."""
        if event_type == EVENT_TYPE_ANY:
            return self.allow_any
        elif event_type in self.schema:
            return True
        elif event_type == EVENT_TYPE_GENERIC:
            return self.allow_generic
        elif not self.schema:
            return True
        else:
            return False

    @classmethod
    def _merge_subschema(cls, a, b):
        """Merge two subschemas together recursively."""
        if a is None:
            return b
        elif b is None:
            return a
        if a == MIXED_TYPES or b == MIXED_TYPES:
            return MIXED_TYPES
        elif is_string(a) and is_string(b):
            if a != b:
                return MIXED_TYPES
            return a
        elif type(a) != type(b):
            return MIXED_TYPES
        elif isinstance(a, list):
            if not a:
                return b
            elif not b:
                return a

            strings_a = [s for s in a if is_string(s)]
            strings_b = [s for s in b if is_string(s)]
            nested_a = [s for s in b if not is_string(s)]
            nested_b = [s for s in b if not is_string(s)]

            # Too complicated
            if (strings_a or strings_b) and (nested_a or nested_b):
                return []
            elif strings_a:
                return list(sorted(set(strings_a).union(set(strings_b))))
            elif len(nested_a) == 1 and len(nested_b) == 1:
                return [cls._merge_subschema(nested_a[0], nested_b[0])]
            else:
                return [MIXED_TYPES]

        elif isinstance(a, dict):
            common_keys = set(a).union(set(b))
            return {k: cls._merge_subschema(a.get(k), b.get(k)) for k in common_keys}
        else:
            return MIXED_TYPES

    def merge(self, other):  # type: (Schema) -> Schema
        """Merge a schema (non-recursively) on to an existing one."""
        # prefer the keys of the original over the added one
        empty_schemas = not all(other.schema.values())
        full_schema = {event: s.copy() for event, s in other.schema.items()}
        for event_type, event_schema in self.schema.items():
            full_schema.setdefault(event_type, {})
            full_schema[event_type].update(event_schema)
        return Schema(full_schema,
                      allow_generic=self.allow_generic or other.allow_generic,
                      allow_any=self.allow_any or other.allow_any,
                      allow_missing=(self.allow_missing or other.allow_missing) or empty_schemas)

    def flatten(self):  # type: () -> Schema
        """Flatten a schema to a single event type."""
        flattened = {}
        empty_schemas = not all(self.schema.values())
        for event_type, event_schema in sorted(self.schema.items()):
            flattened.update(event_schema)
        return Schema({EVENT_TYPE_GENERIC: flattened},
                      allow_generic=False,
                      allow_any=True,
                      allow_missing=self.allow_missing or empty_schemas)

    @classmethod
    def default(cls, default=None):  # type: (Schema) -> Schema
        """Retrieve the active schema or the default."""
        if default is not None:
            cls._default_schema = default
        return cls._default_schema

    @classmethod
    def current(cls):  # type: () -> Schema
        """Retrieve the active schema or the default."""
        current = cls.read_stack("schema")
        if current is None:
            return cls.default()
        return current

    @classmethod
    def _get_item_schema(cls, data):
        """Get the schema for an event."""
        if isinstance(data, dict):
            schema = {}
            for k, v in data.items():
                s = cls._get_item_schema(v)
                if IDENT_RE.match(k) and s is not None:
                    schema[k] = s
            return schema

        if data is None:
            return TypeHint.Null.value
        elif isinstance(data, list):
            schema_base = set()
            nested_schema = None
            for v in data:
                s = cls._get_item_schema(v)
                if is_string(s):
                    schema_base.add(s)
                elif s is not None:
                    if nested_schema is not None:
                        cls._merge_subschema(nested_schema, s)
                    else:
                        nested_schema = s

            if nested_schema is not None and schema_base:
                return MIXED_TYPES
            elif schema_base:
                return list(sorted(schema_base))
            else:
                return nested_schema
        elif is_string(data):
            return TypeHint.String.value
        elif isinstance(data, bool):
            return TypeHint.Boolean.value
        elif is_number(data):
            return TypeHint.Numeric.value

    @classmethod
    def learn(cls, events):
        """Learn the active schema for a list of events."""
        from .events import Event
        schema = {}
        allow_generic = False
        for event in events:
            if not isinstance(event, Event):
                event = Event.from_data(event)
            if event.type == EVENT_TYPE_GENERIC:
                allow_generic = True
            item_schema = cls._get_item_schema(event.data)
            schema[event.type] = cls._merge_subschema(schema.get(event.type), item_schema)
        return Schema(schema, allow_generic=allow_generic)


EMPTY_SCHEMA = Schema({}, allow_generic=True, allow_any=True)

Schema.default(EMPTY_SCHEMA)
