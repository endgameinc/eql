"""Utility functions for testing."""

import random
import string

from eql.ast import Expression, Literal
from eql.parser import parse_expression
from eql.utils import is_string


MAX_INT = int(2 ^ 63 - 1)
MIN_INT = int(- (2 ^ 63))

LETTERS = string.printable


def fold(expr):
    """Test method for parsing and folding."""
    if is_string(expr):
        expr = parse_expression(expr)
        return expr.fold()
    elif isinstance(expr, Expression):
        return expr.fold()
    else:
        raise TypeError("Unable to fold {}".format(expr))


def unfold(value):
    """Convert a python constant into an unparsed EQL expression."""
    return Literal.from_python(value).render()


def random_int():
    """Generate a random integer value."""
    return random.randint(MIN_INT, MAX_INT)


def random_string(max_len=64):
    """Generate a random string."""
    return ''.join(random.choice(LETTERS) for _ in range(max_len))


def random_bool():
    """Choose between ``True`` and ``False``."""
    return random.choice([True, False])


def random_float(scale=100000):
    """Generate a random float within ``scale`` of a maximum range."""
    return random.uniform(-scale, +scale)


def random_value():
    """Choose a random value of a random type."""
    return random.choice(RANDOM_GETTERS)()


RANDOM_GETTERS = [random_int, random_string, random_bool, random_float]
