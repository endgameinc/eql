"""Eventing data schemas."""
from eql.etc import get_etc_path
from eql.utils import load_dump
import contextlib


SCHEMA_FILE = get_etc_path('schema.json')
_schema = {}

EVENT_TYPE_ANY = 'any'
EVENT_TYPE_GENERIC = 'generic'


def reset_schema():
    """Reset the schema to the default."""
    global _schema
    update_schema(load_dump(SCHEMA_FILE))


def check_event_name(name):
    """Check if an event is recognized by the schema."""
    return name in (EVENT_TYPE_ANY, EVENT_TYPE_GENERIC) or name in _schema['event_types']


def update_schema(schema):
    """Update the eventing schema."""
    _schema.clear()
    _schema.update(schema)


@contextlib.contextmanager
def use_schema(schema=None):
    """Context manager for using python's `with` syntax for using a schema when parsing."""
    current_schema = _schema.copy()
    if schema is not None:
        try:
            update_schema(schema)
            yield
        finally:
            update_schema(current_schema)

    else:
        yield


reset_schema()
