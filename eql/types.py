"""EQL type system."""

BASE_STRING = "string"
BASE_NUMBER = "number"
BASE_BOOLEAN = "boolean"
BASE_NULL = "null"
BASE_STRICT_PRIMITIVES = BASE_STRING, BASE_NUMBER, BASE_BOOLEAN
BASE_PRIMITIVES = BASE_STRING, BASE_NUMBER, BASE_BOOLEAN, BASE_NULL

VARIABLE = "variable"

LITERAL_SPECIFIER = "literal"
DYNAMIC_SPECIFIER = "dynamic"
NO_SPECIFIER = "none"
SPECIFIERS = (LITERAL_SPECIFIER, DYNAMIC_SPECIFIER, NO_SPECIFIER)

STRING = NO_SPECIFIER, BASE_STRING
NUMBER = NO_SPECIFIER, BASE_NUMBER
BOOLEAN = NO_SPECIFIER, BASE_BOOLEAN
NULL = NO_SPECIFIER, BASE_NULL
PRIMITIVES = NO_SPECIFIER, BASE_PRIMITIVES


class Array(tuple):
    """Array of nested types."""

    def __repr__(self):
        """Representation string of the object."""
        return type(self).__name__ + tuple.__repr__(self)


class Nested(tuple):
    """Schema of nested types."""

    def subschema(self, name):
        """Get the subschema, given a subfield."""
        for (key, value) in self:
            if name == key:
                return value

        if not self:
            return BASE_ALL

    def __repr__(self):
        """Representation string of the object."""
        return type(self).__name__ + tuple.__repr__(self)


def split(type_hint):
    """Split the specifier from the type hint."""
    if isinstance(type_hint, tuple) and len(type_hint) == 2:
        if type_hint[0] in SPECIFIERS:
            return type_hint

    # Create one if it's not present
    return NO_SPECIFIER, type_hint


def get_specifier(type_hint):
    """Get only the specifier from a type hint."""
    spec, _ = split(type_hint)
    return spec


def get_type(type_hint):
    """Get only the type portion of the type hint."""
    _, hint = split(type_hint)
    return hint


def dynamic(type_hint=None):
    """Make a type hint dynamic."""
    if type_hint is None:
        return DYNAMIC_SPECIFIER, BASE_ALL
    return DYNAMIC_SPECIFIER, get_type(type_hint)


def literal(type_hint=None):
    """Make a type hint literal."""
    if type_hint is None:
        return LITERAL_SPECIFIER, BASE_ALL
    return LITERAL_SPECIFIER, get_type(type_hint)


def clear(type_hint=None):
    """Make a type hint literal."""
    if type_hint is None:
        return NO_SPECIFIER, BASE_ALL
    return NO_SPECIFIER, get_type(type_hint)


# Create a union of all of the full types
ARRAY = NO_SPECIFIER, Array()
PRIMITIVE_ARRAY = NO_SPECIFIER, Array(BASE_PRIMITIVES)
BASE_ALL = (BASE_STRING, BASE_NUMBER, BASE_BOOLEAN, BASE_NULL, Array(), Nested())
EXPRESSION = NO_SPECIFIER, BASE_ALL


def union_specifiers(*specifiers):
    """Union multiple hints together."""
    if DYNAMIC_SPECIFIER in specifiers:
        return DYNAMIC_SPECIFIER

    # literals can't be unioned with other literals
    return NO_SPECIFIER


def _flatten(v):
    if is_union(v):
        for v1 in v:
            for v2 in _flatten(v1):
                yield v2
    else:
        yield v


def union_types(*base_hints):
    """Union multiple type hints together."""
    base_hints = tuple(set(v for v in _flatten(base_hints)))

    if len(base_hints) == 1:
        return base_hints[0]
    return base_hints


def intersect_types(*base_hints):
    """Intersect multiple type hints together."""
    base_hints = tuple(set(v for v in _flatten(base_hints)))

    if len(base_hints) == 1:
        return base_hints[0]
    return base_hints


def union(*type_hints):
    """Union multiple hints together."""
    specifiers, base_hints = zip(*map(split, type_hints))
    return union_specifiers(*specifiers), union_types(*base_hints)


def is_union(type_hint):
    """Determine if a type hint is a union of multiple types."""
    return isinstance(type_hint, tuple) and not isinstance(type_hint, (Array, Nested))


def is_dynamic(type_hint):
    """Check if a type hint is dynamic."""
    return get_specifier(type_hint) == DYNAMIC_SPECIFIER


def is_literal(type_hint):
    """Check if a type hint is dynamic."""
    return get_specifier(type_hint) == LITERAL_SPECIFIER


def check_specifiers(expected_specifier, actual_specifier):
    """Check that specifiers are satisfied."""
    if expected_specifier == NO_SPECIFIER:
        return True
    return expected_specifier == actual_specifier


def check_full_hint(expected_hint, actual_hint):
    """Check that specifiers and types match."""
    expected_spec, expected_type = split(expected_hint)
    actual_spec, actual_type = split(actual_hint)
    return check_specifiers(expected_spec, actual_spec) and check_types(expected_type, actual_type)


def check_types(expected_type, actual_type):
    """Asymmetric check if a type can be matched against another."""
    expected_type = get_type(expected_type)
    actual_type = get_type(actual_type)
    status = _check_types(expected_type, actual_type)

    return status


def _check_types(expected_type, actual_type):
    if expected_type is BASE_ALL or actual_type is BASE_ALL:
        return True

    if is_union(expected_type):
        return any(_check_types(exp, actual_type) for exp in expected_type)

    if is_union(actual_type):
        return any(_check_types(expected_type, act) for act in actual_type)

    # For two arrays, check that there is an intersection between the two
    if isinstance(expected_type, Array):
        if not isinstance(actual_type, Array):
            return False
        elif len(expected_type) == 0 or len(actual_type) == 0:
            return True
        return _check_types(union_types(*expected_type), union_types(*actual_type))

    if isinstance(expected_type, Nested):
        if not isinstance(actual_type, Nested):
            return False
        elif len(expected_type) == 0 or len(actual_type) == 0:
            return True

        keys1, _ = zip(*expected_type)
        keys2, _ = zip(*actual_type)
        matching_keys = set(keys1) ^ set(keys2)

        # If any of the schemas intersect, then they can be compared
        for key in matching_keys:
            if _check_types(expected_type.subschema(key), actual_type.subschema(key)):
                return True
        return False

    return expected_type == actual_type
