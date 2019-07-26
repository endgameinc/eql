"""EQL functions."""
import re

from .signatures import SignatureMixin
from .errors import EqlError
from .types import (
    STRING, NUMBER, BOOLEAN, VARIABLE, ARRAY, literal, PRIMITIVES, EXPRESSION, PRIMITIVE_ARRAY, is_literal, is_dynamic
)
from .utils import is_string, to_unicode, is_number


_registry = {}


def register(func):
    """Register a function signature."""
    if func.name in _registry:
        raise EqlError(u"Function {func.name} already registered. Unable to register {func}".format(func=func))

    _registry[func.name] = func
    return func


def list_functions():
    """Get a list of all current functions."""
    return list(sorted(_registry))


class FunctionSignature(SignatureMixin):
    """Helper class for declaring function signatures."""

    name = str()
    return_value = BOOLEAN

    @classmethod
    def get_callback(cls, *arguments):
        """Get a callback function for the AST."""
        return cls.run

    @classmethod
    def run(cls, *arguments):
        """Reference implementation of the function."""
        raise NotImplementedError()


def get_function(name):  # type: (str) -> FunctionSignature
    """Find a function in the registry."""
    return _registry.get(name)


# noinspection PyAbstractClass
class DynamicFunctionSignature(FunctionSignature):
    """Function signature that can only be processed in a transpiler."""

    @classmethod
    def get_callback(cls, *arguments):
        """Get a callback function for the AST."""
        raise NotImplementedError("Function {} can only be processed in a transpiler".format(cls.name))


# noinspection PyAbstractClass
class MathFunctionSignature(FunctionSignature):
    """Base signature for math functions."""

    argument_types = [NUMBER, NUMBER]
    return_value = NUMBER


@register
class Add(MathFunctionSignature):
    """Add two numbers together."""

    name = "add"

    @classmethod
    def run(cls, x, y):
        """Add two variables together."""
        if is_number(x) and is_number(y):
            return x + y


@register
class ArrayContains(FunctionSignature):
    """Check if ``value`` is a member of the array ``some_array``."""

    name = "arrayContains"
    argument_types = [PRIMITIVE_ARRAY, PRIMITIVES]
    return_value = BOOLEAN
    additional_types = PRIMITIVES

    @classmethod
    def run(cls, array, *value):
        """Search an array for a literal value."""
        values = [v.lower() if is_string(v) else v for v in value]

        if array is not None:
            for v in array:
                if is_string(v) and v.lower() in values:
                    return True
                elif v in values:
                    return True
        return False


@register
class ArrayCount(DynamicFunctionSignature):
    """Search for matches to a dynamic expression in an array."""

    name = "arrayCount"
    argument_types = [ARRAY, VARIABLE, EXPRESSION]
    return_value = NUMBER


@register
class ArraySearch(DynamicFunctionSignature):
    """Search for matches to a dynamic expression in an array."""

    name = "arraySearch"
    argument_types = [ARRAY, VARIABLE, EXPRESSION]
    return_value = BOOLEAN


@register
class Concat(FunctionSignature):
    """Concatenate multiple values as strings."""

    name = "concat"
    additional_types = PRIMITIVES
    minimum_args = 1
    return_value = STRING

    @classmethod
    def run(cls, *arguments):
        """Concatenate multiple values as strings."""
        output = [to_unicode(arg) for arg in arguments]
        return "".join(output)


@register
class Divide(MathFunctionSignature):
    """Divide numeric values."""

    name = "divide"

    @classmethod
    def run(cls, x, y):
        """Divide numeric values."""
        if is_number(x) and is_number(y):
            return float(x) / float(y)


@register
class EndsWith(FunctionSignature):
    """Check if a string ends with a substring."""

    name = "endsWith"
    argument_types = [STRING, STRING]
    return_value = BOOLEAN

    @classmethod
    def run(cls, source, substring):
        """Check if a string ends with a substring."""
        if is_string(source) and is_string(substring):
            return source.lower().endswith(substring.lower())


@register
class IndexOf(FunctionSignature):
    """Check the start position of a substring."""

    name = "indexOf"
    argument_types = [STRING, STRING, NUMBER]
    return_value = NUMBER
    minimum_args = 2

    @classmethod
    def run(cls, source, substring, start=None):
        """Check the start position of a substring."""
        if start is None:
            start = 0

        if is_string(source) and is_string(substring):
            source = source.lower()
            substring = substring.lower()
            if substring in source[start:]:
                return source.index(substring, start)


@register
class Length(FunctionSignature):
    """Get the length of an array or string."""

    name = "length"
    argument_types = [(STRING, ARRAY)]
    return_value = NUMBER

    @classmethod
    def run(cls, array):
        """Get the length of an array or string."""
        if is_string(array) or isinstance(array, (dict, list, tuple)):
            return len(array)
        return 0


@register
class Match(FunctionSignature):
    """Perform regular expression matching on a string."""

    name = "match"
    argument_types = [STRING, literal(STRING)]
    return_value = BOOLEAN
    additional_types = literal(STRING)

    @classmethod
    def join_regex(cls, *regex):
        """Convert a list of wildcards to a regular expression."""
        return "|".join(regex)

    @classmethod
    def get_callback(cls, source_ast, *regex_literals):
        """Get a callback function that uses the compiled regex."""
        regs = [reg.value for reg in regex_literals]
        compiled = re.compile("|".join(regs), re.IGNORECASE | re.UNICODE)

        def callback(source, *_):
            return is_string(source) and compiled.match(source) is not None

        return callback

    @classmethod
    def validate(cls, arguments, type_hints=None):
        """Validate the calling convention and change the argument order if necessary."""
        # used to have just two arguments and the pattern was on the left and expression on the right
        if len(arguments) == 2 and type_hints and is_literal(type_hints[0]) and is_dynamic(type_hints[1]):
            arguments = list(reversed(arguments))
            type_hints = list(reversed(type_hints))
        return super(Match, cls).validate(arguments, type_hints)

    @classmethod
    def run(cls, source, *matches):
        """Compare a string against a list of wildcards."""
        if isinstance(source, bytes):
            source = source.decode("utf-8", "ignore")

        if is_string(source):
            match = re.match("|".join(matches), source, re.IGNORECASE | re.UNICODE | re.MULTILINE | re.DOTALL)
            return match is not None


@register
class MatchLite(Match):
    """Perform lightweight regular expression matching on a string."""

    name = "matchLite"


@register
class Modulo(MathFunctionSignature):
    """Divide numeric values."""

    name = "modulo"

    @classmethod
    def run(cls, x, y):
        """Divide numeric values."""
        if is_number(x) and is_number(y):
            return x % y


@register
class Multiply(MathFunctionSignature):
    """multiply numeric values."""

    name = "multiply"

    @classmethod
    def run(cls, x, y):
        """Multiply numeric values."""
        if is_number(x) and is_number(y):
            return x * y


@register
class Safe(FunctionSignature):
    """Evaluate an expression and suppress exceptions."""

    name = "safe"
    argument_types = [EXPRESSION]
    return_value = EXPRESSION


@register
class StartsWith(FunctionSignature):
    """Check if a string starts with a substring."""

    name = "startsWith"
    argument_types = [STRING, STRING]
    return_value = BOOLEAN

    @classmethod
    def run(cls, source, substring):
        """Check if a string ends with a substring."""
        if is_string(source) and is_string(substring):
            return source.lower().startswith(substring.lower())


@register
class StringContains(FunctionSignature):
    """Check if a string is a substring of another."""

    name = "stringContains"
    argument_types = [STRING, STRING]
    return_value = BOOLEAN

    @classmethod
    def run(cls, source, substring):
        """Check if a string is a substring of another."""
        if is_string(source) and is_string(substring):
            return substring.lower() in source.lower()
        return False


@register
class Substring(FunctionSignature):
    """Extract a substring."""

    name = "substring"
    argument_types = [STRING, NUMBER, NUMBER]
    return_value = STRING
    minimum_args = 1

    @classmethod
    def run(cls, a, start=None, end=None):
        """Extract a substring."""
        if is_string(a):
            return a[start:end]


@register
class Subtract(MathFunctionSignature):
    """Subtract two numbers."""

    name = "subtract"

    @classmethod
    def run(cls, x, y):
        """Add two variables together."""
        if is_number(x) and is_number(y):
            return x - y


@register
class ToNumber(FunctionSignature):
    """Convert a string to a number."""

    name = "number"
    argument_types = [(STRING, NUMBER), NUMBER]
    return_value = NUMBER
    minimum_args = 1

    @classmethod
    def run(cls, source, base=10):
        """Convert a string to a number."""
        if source is None:
            return 0
        elif is_number(source):
            return source
        elif is_string(source):
            if source.isdigit():
                return int(source, base)
            elif source.startswith("0x"):
                return int(source[2:], 16)
            elif len(source.split(".")) == 2:
                return float(source)


@register
class ToString(FunctionSignature):
    """Convert a value to a string."""

    name = "string"
    argument_types = [PRIMITIVES]
    return_value = STRING

    @classmethod
    def run(cls, source):
        """"Convert a value to a string."""
        return to_unicode(source)


@register
class Wildcard(FunctionSignature):
    """Perform glob matching on a string."""

    name = "wildcard"
    argument_types = [STRING, literal(STRING)]
    return_value = BOOLEAN
    additional_types = literal(STRING)

    @classmethod
    def to_regex(cls, *wildcards):
        """Convert a list of wildcards to a regular expression."""
        expressions = []
        head = "^"
        tail = "$"

        for wildcard in wildcards:
            pieces = [re.escape(p) for p in wildcard.lower().split('*')]
            regex = head + '.*?'.join(pieces) + tail

            tail_skip = '.*?$'

            if regex.endswith(tail_skip):
                regex = regex[:-len(tail_skip)]
            expressions.append(regex)

        return "|".join(expressions)

    @classmethod
    def get_callback(cls, source_ast, *wildcard_literals):
        """Get a callback function that uses the compiled regex."""
        wc_values = [wc.value for wc in wildcard_literals]
        pattern = cls.to_regex(*wc_values)
        compiled = re.compile(pattern, re.IGNORECASE | re.UNICODE)

        def callback(source, *_):
            return is_string(source) and compiled.match(source) is not None

        return callback

    @classmethod
    def run(cls, source, *wildcards):
        """Compare a string against a list of wildcards."""
        pattern = cls.to_regex(*wildcards)
        compiled = re.compile(pattern, re.IGNORECASE | re.UNICODE | re.MULTILINE | re.DOTALL)
        return is_string(source) and compiled.match(source) is not None
