"""EQL functions."""
import re
import socket
import struct

from .signatures import SignatureMixin
from .errors import EqlError
from .types import TypeHint
from .utils import is_string, to_unicode, is_number, fold_case, is_insensitive


_registry = {}
REGEX_FLAGS = re.UNICODE | re.DOTALL
MAX_IP = 0xffffffff


def regex_flags():
    """Helper function to properly cased regex flags."""
    return (REGEX_FLAGS | re.IGNORECASE) if is_insensitive() else REGEX_FLAGS


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
    return_value = TypeHint.Unknown
    sometimes_null = False

    @classmethod
    def get_callback(cls, *arguments):
        """Get a callback function for the AST."""
        return cls.run

    @classmethod
    def optimize(cls, arguments):
        """Optimize each function independently."""
        return FunctionCall(cls.name, arguments)

    @classmethod
    def alternate_render(cls, arguments, precedence=None, **kwargs):
        """Return an alternate rendering for a function."""

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

    argument_types = [TypeHint.Numeric, TypeHint.Numeric]
    return_value = TypeHint.Numeric
    operator = None

    @classmethod
    def optimize(cls, arguments):
        """Convert to a MathOperation."""
        if cls.operator:
            return MathOperation(arguments[0], cls.operator, arguments[1])
        return FunctionCall(cls.name, arguments)


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
    argument_types = [TypeHint.Array, TypeHint.primitives()]
    return_value = TypeHint.Boolean
    additional_types = TypeHint.primitives()

    @classmethod
    def run(cls, array, *value):
        """Search an array for a literal value."""
        values = [fold_case(v) for v in value]

        if array is not None:
            for v in array:
                if fold_case(v) in values:
                    return True
            return False

    @classmethod
    def validate(cls, arguments):  # type: (list[NodeInfo]) -> int
        """Validate that the type of the expression matches the type of the array elements."""
        from .schema import Schema

        invalid = super(ArrayContains, cls).validate(arguments)
        if invalid is not None:
            return invalid

        if isinstance(arguments[0].schema, list) and len(arguments[0].schema) == 1:
            element_type, _ = Schema.convert_to_type(arguments[0].schema[0])
            if not arguments[1].validate(element_type):
                return 1


@register
class ArrayCount(DynamicFunctionSignature):
    """Search for matches to a dynamic expression in an array."""

    name = "arrayCount"
    argument_types = [TypeHint.Array, TypeHint.Variable, TypeHint.Boolean]
    return_value = TypeHint.Numeric


@register
class ArraySearch(DynamicFunctionSignature):
    """Search for matches to a dynamic expression in an array."""

    name = "arraySearch"
    argument_types = [TypeHint.Array, TypeHint.Variable, TypeHint.Boolean]
    return_value = TypeHint.Boolean


@register
class Between(FunctionSignature):
    """Return a substring that's between two other substrings."""

    name = "between"
    argument_types = [TypeHint.String, TypeHint.String, TypeHint.String, TypeHint.Boolean]
    minimum_args = 3
    return_value = TypeHint.String
    sometimes_null = True

    @classmethod
    def run(cls, source_string, first, second, greedy=False):
        """Return the substring between two other ones."""
        if is_string(source_string) and is_string(first) and is_string(second):
            match_string = fold_case(source_string)
            first = fold_case(first)
            second = fold_case(second)

            try:
                start_pos = match_string.index(first) + len(first)
                end_pos = match_string.rindex(second, start_pos) if greedy else match_string.index(second, start_pos)
                return source_string[start_pos:end_pos]

            except ValueError:
                return


@register
class CidrMatch(FunctionSignature):
    """Math an IP address against a list of IPv4 or IPv6 subnets in CIDR notation."""

    name = "cidrMatch"
    argument_types = [TypeHint.String, TypeHint.String.require_literal()]
    additional_types = TypeHint.String.require_literal()
    return_value = TypeHint.Boolean

    octet_re = r'(?:25[0-5]|2[0-4][0-9]|[01]?[0-9]?[0-9])'
    ipv4_re = r'\.'.join([octet_re, octet_re, octet_re, octet_re])
    ipv4_compiled = re.compile(r'^{}$'.format(ipv4_re))
    cidrv4_compiled = re.compile(r'^{}/(?:3[0-2]|2[0-9]|1[0-9]|[0-9])$'.format(ipv4_re))
    
    h16_re = r'[a-fA-F0-9]{1,4}'
    ipv6_re = r'^(?:[a-fA-F0-9]{1,4}:){7}[a-fA-F0-9]{1,4}$'
    ipv6_compiled = re.compile(ipv6_re)
    cidrv6_re = r'(?:[a-fA-F0-9]{1,4}:){0,7}[a-fA-F0-9]{1,4}/(?:12[0-8]|1[01][0-9]|[0-9]{1,2})'
    cidrv6_compiled = re.compile('^{}$'.format(cidrv6_re))

    # store it in native representation, then recover it in network order
    masks4 = [struct.unpack(">L", struct.pack(">L", MAX_IP & ~(MAX_IP >> b)))[0] for b in range(33)]
    mask_addresses4 = [socket.inet_ntoa(struct.pack(">L", m)) for m in masks4]

    masks6 = [int('1' * i + '0' * (128 - i), 2) for i in range(129)]
    mask_addresses6 = [socket.inet_ntop(socket.AF_INET6, struct.pack(">QQ", m >> 64, m & 0xFFFFFFFFFFFFFFFF)) for m in masks6]

    # TODO need to add function to expand IPv6 shorthand addresses to full addresses before matching

    @classmethod
    def to_mask(cls, cidr_string):
        """Split an IP address plus cidr block to the mask."""
        ip_string, size = cidr_string.split("/")
        size = int(size)
        if cls.ipv4_compiled.match(ip_string):
            ip_bytes = socket.inet_aton(ip_string)
            subnet_int, = struct.unpack(">L", ip_bytes)
            mask = cls.masks4[size]
        elif cls.ipv6_compiled.match(ip_string):
            ip_bytes = socket.inet_pton(socket.AF_INET6, ip_string)
            # TODO remove x if possible
            subnet_int, x = struct.unpack(">QQ", ip_bytes)
            mask = cls.masks6[size]
        else:
            raise ValueError("Invalid IP address")

        return subnet_int & mask, mask

    @classmethod
    def make_octet_re(cls, start, end):
        """Convert an octet-range into a regular expression."""
        combos = []

        if start == end:
            return "{:x}".format(start)

        if start == 0 and end == 255:
            return cls.octet_re

        # IPv4 octet range
        if end <= 255:
            for o in range(start, end + 1):
                combos.append("{:d}".format(o))
        # IPv6 h16 range
        elif start >= 0 and end <= 65535:
            for h16 in range(start, end + 1):
                combos.append("{:x}".format(h16))
        else:
            raise ValueError("Invalid octet range")

        return "(?:{})".format("|".join(combos))

    @classmethod
    def make_cidr_regex(cls, cidr):
        """Convert a list of wildcards strings for matching a cidr."""
        if cls.ipv4_compiled.match(cidr):
            min_octets, max_octets = cls.to_range(cidr)
            return r"\.".join(cls.make_octet_re(*pair) for pair in zip(min_octets, max_octets))
        elif cls.ipv6_compiled.match(cidr):
            h16s, prefix_len = cidr.split("/")
            h16s = h16s.split(":")
            prefix_len = int(prefix_len)
            if len(h16s) < 8:
                h16s += [""] * (8 - len(h16s))
            h16s = [int(h, 16) if h else 0 for h in h16s]
            min_h16s = [h & (0xFFFF << (16 - prefix_len)) for h in h16s]
            max_h16s = [h | (0xFFFF >> prefix_len) for h in min_h16s]
            return ":".join(cls.make_octet_re(*pair) for pair in zip(min_h16s, max_h16s))
        else:
            raise ValueError("Invalid CIDR notation")

    @classmethod
    def to_range(cls, cidr):
        """Get the IP range for a list of IP addresses."""
        if cls.ipv4_compiled.match(cidr):
            ip_string, prefix_len = cidr.split("/")
            prefix_len = int(prefix_len)
            ip_integer = struct.unpack(">L", socket.inet_aton(ip_string))[0]
            mask = cls.masks4[prefix_len]
            max_ip_integer = ip_integer | (MAX_IP ^ mask)
            min_octets = struct.unpack("BBBB", struct.pack(">L", ip_integer))
            max_octets = struct.unpack("BBBB", struct.pack(">L", max_ip_integer))
        elif cls.ipv6_compiled.match(cidr):
            ip_string, prefix_len = cidr.split("/")
            prefix_len = int(prefix_len)
            ip_bytes = socket.inet_pton(socket.AF_INET6, ip_string)
            ip_integer = int.from_bytes(ip_bytes, byteorder="big")
            mask = cls.masks6[prefix_len]
            max_ip_integer = ip_integer | (MAX_IP6 ^ mask)
            min_h16s = struct.unpack(">8H", struct.pack(">QQ", ip_integer >> 64, ip_integer & 0xFFFFFFFFFFFFFFFF))
            max_h16s = struct.unpack(">8H", struct.pack(">QQ", max_ip_integer >> 64, max_ip_integer & 0xFFFFFFFFFFFFFFFF))
            min_octets = [h16 >> 8 for h16 in min_h16s] + [h16 & 0xFF for h16 in min_h16s[6:]]
            max_octets = [h16 >> 8 for h16 in max_h16s] + [h16 & 0xFF for h16 in max_h16s[6:]]
        else:
            raise ValueError("Invalid CIDR notation")

        return min_octets, max_octets

    @classmethod
    def get_callback(cls, _, *cidr_matches):
        """Get the callback function with all the masks converted."""
        ipv4_masks = []
        ipv6_masks = []
        for cidr in cidr_matches:
            if cls.ipv4_compiled.match(cidr.value):
                ipv4_masks.append(cls.to_mask(cidr.value))
            elif cls.ipv6_compiled.match(cidr.value):
                ipv6_masks.append(cls.to_mask(cidr.value))

        def callback(source, *_):
            if is_string(source) and (cls.ipv4_compiled.match(source) or cls.ipv6_compiled.match(source)):
                if cls.ipv4_compiled.match(source):
                    ip_integer, _ = cls.to_mask(source + "/32")
                    for subnet, mask in ipv4_masks:
                        if ip_integer & mask == subnet:
                            return True
                elif cls.ipv6_compiled.match(source):
                    ip_integer, _ = cls.to_mask(source + "/128")
                    for subnet, mask in ipv6_masks:
                        if ip_integer & mask == subnet:
                            return True

            return False

        return callback

    @classmethod
    def run(cls, ip_address, *cidr_matches):
        """Compare an IP address against a list of cidr blocks."""
        if is_string(ip_address) and (cls.ipv4_compiled.match(ip_address) or cls.ipv6_compiled.match(ip_address)):
            if cls.ipv4_compiled.match(ip_address):
                ip_integer, _ = cls.to_mask(ip_address + "/32")
                for cidr in cidr_matches:
                    if is_string(cidr) and cls.cidrv4_compiled.match(cidr):
                        subnet, mask = cls.to_mask(cidr)
                        if ip_integer & mask == subnet:
                            return True
            elif cls.ipv6_compiled.match(ip_address):
                ip_integer, _ = cls.to_mask(ip_address + "/128")
                for cidr in cidr_matches:
                    if is_string(cidr) and cls.cidrv6_compiled.match(cidr):
                        subnet, mask = cls.to_mask(cidr)
                        if ip_integer & mask == subnet:
                            return True

        return False

    @classmethod
    def validate(cls, arguments):
        """Validate the calling convention and change the argument order if necessary."""
        # used to have just two arguments and the pattern was on the left and expression on the right
        error_position = super(CidrMatch, cls).validate(arguments)

        if error_position is not None:
            return error_position

        # create a copy of the array that we can modify
        arguments = arguments[:]

        for pos, argument in enumerate(arguments[1:], 1):
            # overwrite the original node
            text = argument.node.value.strip()

            if not (cls.cidrv4_compiled.match(argument.node.value) or cls.cidrv6_compiled.match(argument.node.value)):
                return pos

            # Since it does match, we should also rewrite the string to align to the base of the subnet
            ip_address, size = text.split("/")
            subnet_integer, _ = cls.to_mask(text)
            if cls.cidrv4_compiled.match(argument.node.value):
                subnet_bytes = struct.pack(">L", subnet_integer)
                subnet_base = socket.inet_ntoa(subnet_bytes)
            elif cls.cidrv6_compiled.match(argument.node.value):
                subnet_bytes = struct.pack(">QQ", subnet_integer >> 64, subnet_integer & 0xFFFFFFFFFFFFFFFF)
                subnet_base = socket.inet_ntop(socket.AF_INET6, subnet_bytes)

            # overwrite the original argument so it becomes the subnet
            argument.node = String("{}/{}".format(subnet_base, size))

        return None


@register
class Concat(FunctionSignature):
    """Concatenate multiple values as strings."""

    name = "concat"
    additional_types = TypeHint.primitives()
    minimum_args = 1
    return_value = TypeHint.String

    @classmethod
    def run(cls, *arguments):
        """Concatenate multiple values as strings."""
        if all(arg is not None for arg in arguments):
            output = [ToString.run(arg) for arg in arguments]
            return "".join(output)


@register
class Divide(MathFunctionSignature):
    """Divide numeric values."""

    name = "divide"

    @classmethod
    def run(cls, x, y):
        """Divide numeric values."""
        if is_number(x) and is_number(y) and y != 0:
            if isinstance(x, float) or isinstance(y, float):
                return float(x) / float(y)
            return x // y


@register
class EndsWith(FunctionSignature):
    """Check if a string ends with a substring."""

    name = "endsWith"
    argument_types = [TypeHint.String, TypeHint.String]
    return_value = TypeHint.Boolean

    @classmethod
    def run(cls, source, substring):
        """Check if a string ends with a substring."""
        if is_string(source) and is_string(substring):
            return fold_case(source).endswith(fold_case(substring))


@register
class IndexOf(FunctionSignature):
    """Check the start position of a substring."""

    name = "indexOf"
    argument_types = [TypeHint.String, TypeHint.String, TypeHint.Numeric]
    return_value = TypeHint.Numeric
    sometimes_null = True
    minimum_args = 2

    @classmethod
    def run(cls, source, substring, start=None):
        """Check the start position of a substring."""
        if start is None:
            start = 0

        if is_string(source) and is_string(substring):
            source = fold_case(source)
            substring = fold_case(substring)
            if substring in source[start:]:
                return source.index(substring, start)


@register
class Length(FunctionSignature):
    """Get the length of an array or string."""

    name = "length"
    argument_types = [(TypeHint.String, TypeHint.Array)]
    return_value = TypeHint.Numeric

    @classmethod
    def run(cls, array):
        """Get the length of an array or string."""
        if is_string(array) or isinstance(array, (dict, list, tuple)):
            return len(array)


@register
class Match(FunctionSignature):
    """Perform regular expression matching on a string."""

    name = "match"
    argument_types = [TypeHint.String, TypeHint.String.require_literal()]
    return_value = TypeHint.Boolean
    additional_types = TypeHint.String.require_literal()

    @classmethod
    def join_regex(cls, *regex):
        """Convert a list of wildcards to a regular expression."""
        return "|".join(regex)

    @classmethod
    def get_callback(cls, source_ast, *regex_literals):
        """Get a callback function that uses the compiled regex."""
        regs = [reg.value for reg in regex_literals]
        compiled = re.compile("|".join(regs), regex_flags())

        def callback(source, *_):
            if is_string(source):
                return compiled.match(source) is not None

        return callback

    @classmethod
    def validate(cls, arguments):
        """Validate the calling convention and change the argument order if necessary."""
        if len(arguments) == 2 and isinstance(arguments[0].node, String) and not isinstance(arguments[1].node, String):
            # used to have just two arguments and the pattern was on the left and expression on the right
            arguments[0], arguments[1] = arguments[1], arguments[0]

        pos = super(Match, cls).validate(arguments)

        if pos is not None:
            return pos

        for pos, argument in enumerate(arguments[1:], 1):
            try:
                re.compile(argument.node.value, regex_flags())
            except re.error:
                return pos

    @classmethod
    def run(cls, source, *matches):
        """Compare a string against a list of wildcards."""
        if isinstance(source, bytes):
            source = source.decode("utf-8", "ignore")

        if is_string(source):
            match = re.match("|".join(matches), source, regex_flags())
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
class Safe(DynamicFunctionSignature):
    """Evaluate an expression and suppress exceptions."""

    name = "safe"
    argument_types = [TypeHint.Unknown]
    return_value = TypeHint.Unknown


@register
class StartsWith(FunctionSignature):
    """Check if a string starts with a substring."""

    name = "startsWith"
    argument_types = [TypeHint.String, TypeHint.String]
    return_value = TypeHint.Boolean

    @classmethod
    def run(cls, source, substring):
        """Check if a string ends with a substring."""
        if is_string(source) and is_string(substring):
            return fold_case(source).startswith(fold_case(substring))


@register
class StringContains(FunctionSignature):
    """Check if a string is a substring of another."""

    name = "stringContains"
    argument_types = [TypeHint.String, TypeHint.String]
    return_value = TypeHint.Boolean

    @classmethod
    def run(cls, source, substring):
        """Check if a string is a substring of another."""
        if is_string(source) and is_string(substring):
            return fold_case(substring) in fold_case(source)
        return False


@register
class Substring(FunctionSignature):
    """Extract a substring."""

    name = "substring"
    argument_types = [TypeHint.String, TypeHint.Numeric, TypeHint.Numeric]
    return_value = TypeHint.String
    minimum_args = 2

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
    argument_types = [TypeHint.String, TypeHint.Numeric]
    minimum_args = 1
    return_value = TypeHint.Numeric
    sometimes_null = True

    @classmethod
    def run(cls, source, base=None):
        """Convert a string to a number."""
        if is_string(source):
            if len(source.split(".")) == 2 and base in (None, 10):
                return float(source)
            elif source.startswith("0x") and base in (None, 16):
                return int(source[2:], 16)
            elif source.lstrip("-+").isdigit():
                return int(source, base or 10)


@register
class ToString(FunctionSignature):
    """Convert a value to a string."""

    name = "string"
    argument_types = [TypeHint.primitives()]
    return_value = TypeHint.String

    @classmethod
    def run(cls, source):
        """"Convert a value to a string."""
        if source is not None:
            if isinstance(source, bool):
                return "true" if source else "false"

            return to_unicode(source)


@register
class Wildcard(FunctionSignature):
    """Perform glob matching on a string."""

    name = "wildcard"
    argument_types = [TypeHint.String, TypeHint.String.require_literal()]
    return_value = TypeHint.Boolean
    additional_types = TypeHint.String.require_literal()

    @classmethod
    def to_regex(cls, *wildcards):
        """Convert a list of wildcards to a regular expression."""
        expressions = []
        head = "^"
        tail = "$"

        for wildcard in wildcards:
            pieces = [re.escape(p) for p in fold_case(wildcard).split('*')]
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
        compiled = re.compile(pattern, regex_flags())

        def callback(source, *_):
            if is_string(source):
                return compiled.match(source) is not None

        return callback

    @classmethod
    def alternate_render(cls, arguments, precedence=None, **kwargs):
        """Allow some functions to be rendered back as shorthand."""
        if len(arguments) == 2 and isinstance(arguments[1], String):
            lhs, rhs = arguments
            return Comparison(lhs, Comparison.EQ, rhs).render(precedence, **kwargs)

    @classmethod
    def run(cls, source, *wildcards):
        """Compare a string against a list of wildcards."""
        if is_string(source):
            pattern = cls.to_regex(*wildcards)
            compiled = re.compile(pattern, regex_flags())
            return compiled.match(source) is not None


# circular dependency
from .ast import MathOperation, FunctionCall, Comparison, String  # noqa: E402
