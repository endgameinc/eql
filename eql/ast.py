"""EQL syntax tree nodes/schema."""
from __future__ import unicode_literals

import re
from collections import OrderedDict
from operator import lt, le, eq, ne, ge, gt
from string import Template
from enum import Enum

from .errors import EqlError, EqlCompileError
from .signatures import SignatureMixin
from .types import TypeHint, NodeInfo  # noqa: F401
from .utils import to_unicode, is_string, is_number, ParserConfig, fold_case, chr_compat

__all__ = (
    # base classes
    "BaseNode",
    "Expression",
    "EqlNode",

    # Literals
    "Literal",
    "String",
    "Number",
    "Null",
    "Boolean",
    "TimeRange",
    "TimeUnit",

    # fields and subfields
    "Field",

    # boolean logic
    "Comparison",
    "InSet",
    "IsNotNull",
    "IsNull",
    "And",
    "Or",
    "Not",
    "FunctionCall",
    "MathOperation",

    # queries
    "EventQuery",
    "NamedSubquery",
    "NamedParams",
    "SubqueryBy",
    "Join",
    "Sequence",

    # pipes
    "PipeCommand",

    # full queries
    "PipedQuery",
    "EqlAnalytic",

    # macros
    "Definition",
    "BaseMacro",
    "CustomMacro",
    "Macro",
    "Constant",
    "PreProcessor",
)


class BaseNode(object):
    """This is the base class for all AST nodes."""

    __slots__ = ()

    template = None  # type: Template
    delims = {}
    precedence = None

    def iter_slots(self):
        # type: () -> list
        """Enumerate over all of the slots and their values."""
        for key in self.__slots__:
            yield key, getattr(self, key, None)

    def children(self):  # type: () -> list[BaseNode]
        """Collect over all child nodes."""
        children = []

        def recurse(descendant):
            if isinstance(descendant, BaseNode):
                children.append(descendant)
            elif hasattr(descendant, "__iter__"):
                for c in descendant:
                    recurse(c)

        recurse(c for c in self.iter_slots())
        return children

    def optimize(self, recursive=False):
        """Optimize an AST."""
        return Optimizer(recursive=recursive).walk(self)

    def __eq__(self, other):
        """Check if two ASTs are equivalent."""
        return type(self) == type(other) and list(self.iter_slots()) == list(other.iter_slots())

    def __ne__(self, other):
        """Check if two ASTs are not equivalent."""
        return not self == other

    def render(self, precedence=None, **kwargs):
        """Render the AST in the target language."""
        if not self.template:
            raise NotImplementedError()

        dicted = {}
        for name, value in self.iter_slots():
            if isinstance(value, (list, tuple)):
                delim = self.delims[name]
                value = [v.render(self.precedence, **kwargs) if isinstance(v, BaseNode) else v for v in value]
                value = delim.join(v for v in value)
            elif isinstance(value, BaseNode):
                value = value.render(self.precedence, **kwargs)
            dicted[name] = value
        return self.template.substitute(dicted)

    def __repr__(self):
        """Python representation of the AST."""
        return "{}({})".format(type(self).__name__, ", ".join('{}={}'.format(name, repr(slot))
                                                              for name, slot in self.iter_slots()))

    def __iter__(self):
        """Iterate recursively through all nodes in the tree."""
        return Walker().iter_node(self)

    def __unicode__(self):
        """Render the AST back as a valid EQL string."""
        return self.render()

    def __str__(self):
        """Render the AST back as a valid EQL string."""
        unicoded = self.__unicode__()
        # Python 2.7
        if not isinstance(unicoded, str):
            unicoded = unicoded.encode('utf-8')
        return unicoded


# noinspection PyAbstractClass
class EqlNode(BaseNode):
    """The base class for all nodes within the event query language."""

    TAB = '  '
    precedence = None

    def indent(self, text, depth=1):
        """Indent by EQL default tab."""
        delim = self.TAB * depth
        return '\n'.join(delim + line.rstrip() for line in text.splitlines())

    def _render(self):
        # Render the template if defined
        return super(EqlNode, self).render()

    def render(self, precedence=None, **kwargs):
        """Render an EQL node and add parentheses to support orders of operation."""
        rendered = self._render(**kwargs)
        if precedence is not None and self.precedence is not None and self.precedence > precedence:
            return '({})'.format(rendered)
        return rendered


# noinspection PyAbstractClass
class Expression(EqlNode):
    """Base class for expressions."""

    precedence = 0

    def foldable(self):
        """Determine if an expression can be folded."""
        return all(c.foldable() for c in self.children())

    def fold(self):
        """Fold an expression."""
        optimized = self.optimize(recursive=True)
        if isinstance(optimized, Literal):
            return optimized.value

        raise EqlError("Unable to fold expression: {}".format(self))

    def __and__(self, other):
        """Boolean AND between two AST nodes."""
        if self == Boolean(False) or other == Boolean(False):
            return Boolean(False)
        elif self == Boolean(True):
            return other
        elif other == Boolean(True):
            return self

        if self == Null() or other == Null():
            return Null()

        # flatten out the terms in the And
        terms = []
        for t in (self, other):
            if isinstance(t, And):
                terms.extend(t.terms)
            else:
                terms.append(t)
        return And(terms)

    def __or__(self, other):
        """"Boolean OR between two AST nodes."""
        if self == Boolean(True) or other == Boolean(True):
            return Boolean(True)
        elif self == Boolean(False):
            return other
        elif other == Boolean(False):
            return self

        if self == Null() or other == Null():
            return Null()

        # flatten out the terms in the Or
        terms = []
        for t in (self, other):
            if isinstance(t, Or):
                terms.extend(t.terms)
            else:
                terms.append(t)
        return Or(terms)

    def __invert__(self):
        """Negate an expression with Not."""
        return Not(self)


class Literal(Expression):
    """Static value."""

    __slots__ = 'value',
    precedence = Expression.precedence + 1
    type_hint = None  # type: TypeHint

    def __init__(self, value):
        """Create an EQL value from a python value."""
        if type(self) is Literal:
            raise TypeError("Literal AST nodes can't be created directly. Try Literal.from_python")
        self.value = value

    @classmethod
    def find_type(cls, python_value):
        """Find the corresponding AST node type for a python value."""
        if python_value is None:
            return Null
        elif python_value is True or python_value is False:
            return Boolean
        elif is_number(python_value):
            return Number
        elif is_string(python_value):
            return String
        else:
            raise TypeError("Unable to convert {} to a literal.".format(python_value))

    @classmethod
    def from_python(cls, python_value):
        """Convert a python value to a literal."""
        subcls = cls.find_type(python_value)
        return subcls(python_value)

    def __invert__(self):
        """Negate a static value."""
        return Boolean(not self.value)


class Boolean(Literal):
    """Boolean literal."""

    type_hint = TypeHint.Boolean

    def _render(self):
        return 'true' if self.value else 'false'


class Null(Literal):
    """Null literal."""

    type_hint = TypeHint.Null

    def __init__(self, value=None):
        """Null literal value."""
        super(Null, self).__init__(None)

    def __invert__(self):
        """Null values can't be inverted."""
        return Null()

    def _render(self):
        return 'null'


class Number(Literal):
    """Numeric literal."""

    type_hint = TypeHint.Numeric

    def _render(self):
        return to_unicode(self.value)


class String(Literal):
    """String literal."""

    escape_patterns = {
        '\\': '\\\\',
        '\b': '\\b',
        '\t': '\\t',
        '\r': '\\r',
        '\n': '\\n',
        '\f': '\\f',
        '\"': '\\\"',
        '\'': '\\\'',
    }
    reverse_patterns = {v: k for k, v in escape_patterns.items()}
    escape_re = r'[{}]'.format('|'.join(escape_patterns.values()))
    type_hint = TypeHint.String

    @classmethod
    def escape(cls, s):
        """Escape known patterns in a string."""
        def replace_callback(sub):
            return cls.escape_patterns.get(sub.group(), sub.group())
        return re.sub(cls.escape_re, replace_callback, s)

    @classmethod
    def unescape(cls, s):
        """Unescape an EQL rendered string."""
        def replace_callback(sub):
            original = sub.group()
            if original.startswith("\\u{"):
                hex_digits = original[3:-1]
                as_hex = int(hex_digits, 16)
                return chr_compat(as_hex)
            return cls.reverse_patterns[sub.group()]
        return re.sub(r"\\u\{.+\}|\\[^u]", replace_callback, s)

    def _render(self):
        return '"{}"'.format(self.escape(self.value))


class TimeUnit(Enum):
    """Elasticsearch compatible time units."""

    Milliseconds = "ms"
    Seconds = "s"
    Minutes = "m"
    Hours = "h"
    Days = "d"

    @classmethod
    def get_units(cls):
        """Return all time units in order of precision."""
        return [cls.Milliseconds, cls.Seconds, cls.Minutes, cls.Hours, cls.Days]

    def as_milliseconds(self):
        """Convert a time unit to a number of milliseconds."""
        if self == TimeUnit.Milliseconds:
            return 1
        elif self == TimeUnit.Seconds:
            return 1000
        elif self == TimeUnit.Minutes:
            return 1000 * 60
        elif self == TimeUnit.Hours:
            return 1000 * 60 * 60
        elif self == TimeUnit.Days:
            return 1000 * 60 * 60 * 24
        else:
            raise ValueError("Unknown time unit {}".format(repr(self)))

    def __repr__(self):
        """Override the default __repr__ method."""
        return "{}({})".format(type(self).__name__, repr(self.value))


class TimeRange(Expression):
    """EQL node for an interval of time."""

    __slots__ = 'quantity', 'unit',
    precedence = Expression.precedence + 1

    def __init__(self, quantity, unit):  # type: (int, TimeUnit) -> None
        """EQL time interval."""
        self.quantity = quantity
        self.unit = unit

    def as_milliseconds(self):
        """Convert a time range to milliseconds."""
        return self.quantity * self.unit.as_milliseconds()

    def _render(self):
        return '{:d}{:s}'.format(self.quantity, self.unit.value)


class Field(Expression):
    """Variables and paths in scope of the event."""

    EVENTS = 'events'

    __slots__ = 'base', 'path',
    precedence = Expression.precedence + 1

    field_re = re.compile("^[_A-Za-z][_A-Za-z0-9]*$")

    def __init__(self, base, path=None, as_var=False):
        """Query the event for the field expression.

        :param str base: The root field
        :param list[str|int] path: The sub fields and array positions
        :param bool as_var: Render the node as a variable
        """
        self.base = base
        self.path = path or []
        self.as_var = as_var

    def query_multiple_events(self):  # type: () -> (int, Field)
        """Get the index into the event array and query."""
        if self.base == Field.EVENTS and len(self.path) >= 2:
            if is_number(self.path[0]) and is_string(self.path[1]):
                return self.path[0], Field(self.path[1], self.path[2:])
        return 0, self

    @property
    def full_path(self):  # type: () -> list[str]
        """Get the full path for a field."""
        return [self.base] + self.path

    @classmethod
    def escape_ident(cls, key):
        """Escape identifiers that are keywords."""
        from .parser import keywords

        if key in keywords or cls.field_re.match(key) is None:
            return "`{key}`".format(key=key)
        return key

    def _render(self):
        text = []
        if self.as_var:
            text.append("$")
        text.append(self.escape_ident(self.base))

        for key in self.path:
            if is_number(key):
                text.append("[{}]".format(key))
            else:
                text.append(".{}".format(self.escape_ident(key)))
        return "".join(text)


class FunctionCall(Expression):
    """A call into a user-defined function by name and a list of arguments."""

    __slots__ = 'name', 'arguments', 'as_method'
    precedence = Literal.precedence + 1
    template = Template('$name($arguments)')
    delims = {'arguments': ', '}

    def __init__(self, name, arguments, as_method=False):
        """Call the function by name.

        :param str name: The name of the user-defined function
        :param list[Expression] arguments: Arguments to pass into the function.
        """
        self.name = name
        self.arguments = arguments or []
        self.as_method = as_method

    @property
    def callback(self):
        """Get the callback for this node."""
        return self.signature.get_callback(*self.arguments)

    @property
    def signature(self):
        """Get the matching function signature."""
        return get_function(self.name)

    def _render(self):
        """Determine the precedence by checking if it's called as a method."""
        if self.as_method:
            return '{base}:{name}({remaining})'.format(
                base=self.arguments[0].render(self.precedence), name=self.name,
                remaining=", ".join(arg.render(self.precedence) for arg in self.arguments[1:]))

        return super(FunctionCall, self)._render()

    def render(self, precedence=None):
        """Convert wildcards back to the short hand syntax."""
        if self.signature:
            alternate_render = self.signature.alternate_render(self.arguments, precedence)
            if alternate_render:
                return alternate_render

        return super(FunctionCall, self).render()

    def __or__(self, other):
        """Optimize OR comparisons between matching variadic binary functions."""
        if isinstance(other, FunctionCall) and \
                self.name in ("wildcard", "match", "matchLite") and other.name == self.name:
            # wildcard(x, ABC...) or wildcard(x, DEF...) ==> wildcard(x, ABC..., DEF...)
            if self.arguments[0] == other.arguments[0]:
                return FunctionCall(self.name, self.arguments + other.arguments[1:])

        return super(FunctionCall, self).__or__(other)


class NamedSubquery(Expression):
    """Named of queries perform a subquery with a specific type and returns true if the current event is related.

    Query Types:
    - descendant: Returns true if the pid/unique_pid of the event is a descendant of the subquery process
    - child: Returns true if the pid/unique_pid of the event is a child of the subquery process
    - event: Returns true if the pid/unique_pid of the event matches the subquery process
    """

    __slots__ = 'query_type', 'query'
    precedence = FunctionCall.precedence

    DESCENDANT = 'descendant'
    EVENT = 'event'
    CHILD = 'child'

    supported_types = (DESCENDANT, EVENT, CHILD)
    template = Template('$query_type of [$query]')

    def __init__(self, query_type, query):
        """Init.

        :param str query_type: The type of subquery to relate by
        :param EventQuery query: Query applied to the process' ancestor(s)
        """
        self.query_type = query_type
        self.query = query


class MathOperation(Expression):
    """Mathematical operation between two numeric values."""

    __slots__ = 'left', 'operator', 'right'
    OPERATORS = ('*', '/', '%', '+', '-')

    func_lookup = {"*": "multiply", "+": "add", "-": "subtract", "%": "modulo", "/": "divide"}

    min_precedence = NamedSubquery.precedence + 1
    max_precedence = min_precedence + 1
    full_template = Template('$left $operator $right')
    negative_template = Template('$operator$right')

    def __init__(self, left, operator, right):  # type: (Expression, str, Expression) -> None
        """Mathematical operation between two numeric values."""
        self.left = left
        self.operator = operator
        self.right = right

    def to_function_call(self):
        """Convert a math operator to an EQL function call."""
        return FunctionCall(self.func_lookup[self.operator], [self.left, self.right])

    @property
    def precedence(self):
        """Get the precedence depending on the operator."""
        if self.operator in "*/%":
            return self.min_precedence
        else:
            return self.max_precedence

    @property
    def template(self):
        """Make the template dynamic."""
        return self.negative_template if self.left == Number(0) else self.full_template


class Comparison(Expression):
    """Represents a comparison between two values, as in ``<expr> <comparator> <expr>``.

    Comparison operators include ``==``, ``!=``, ``<``, ``<=``, ``>=``, and ``>``.
    """

    __slots__ = 'left', 'comparator', 'right'
    LT, LE, EQ, NE, GE, GT = ('<', '<=', '==', '!=', '>=', '>')

    func_lookup = {LT: lt, LE: le, EQ: eq, NE: ne, GE: ge, GT: gt}
    precedence = MathOperation.max_precedence + 1
    template = Template('$left $comparator $right')

    def __init__(self, left, comparator, right):
        # type: (Expression, str, Expression) -> None
        """Compare two fields or values to each other."""
        self.left = left
        self.comparator = comparator
        self.right = right
        self.function = self.func_lookup[comparator]

    def __invert__(self):
        """Convert a comparison by flipping the operators."""
        if self.comparator == self.EQ:
            return Comparison(self.left, Comparison.NE, self.right).optimize()
        elif self.comparator == self.NE:
            return Comparison(self.left, Comparison.EQ, self.right).optimize()
        return super(Comparison, self).__invert__()

    def __or__(self, other):
        """Check for one field being compared to multiple values, and switch to a set."""
        if self.comparator == Comparison.EQ and isinstance(self.right, Literal):
            if isinstance(other, Comparison) and self.left == other.left and other.comparator == Comparison.EQ:
                if isinstance(other.right, Literal):
                    return InSet(self.left, [self.right, other.right])
            elif isinstance(other, InSet) and self.left == other.expression and other.is_literal():
                container = [self.right]
                container.extend(other.container)
                return InSet(self.left, container)
        return super(Comparison, self).__or__(other)

    def __and__(self, other):
        """Check if a comparison is ANDed to a set."""
        if self.comparator == Comparison.EQ and isinstance(other, InSet) and self.left == other.expression:
            return InSet(self.left, [self.right]) & other
        return super(Comparison, self).__and__(other)


class IsNull(Expression):
    """Node for checking if values are null."""

    template = Template("$expr == null")
    precedence = Comparison.precedence
    __slots__ = "expr",

    def __init__(self, expr):  # type: (Expression) -> None
        """Check if a value is null."""
        self.expr = expr


class IsNotNull(Expression):
    """Node for checking if values are null."""

    template = Template("$expr != null")
    precedence = Comparison.precedence
    __slots__ = "expr",

    def __init__(self, expr):  # type: (Expression) -> None
        """Check if a value is not null."""
        self.expr = expr


class InSet(Expression):
    """Check if the value of a field within an event matches a list of values."""

    __slots__ = 'expression', 'container'
    precedence = Comparison.precedence

    def __init__(self, expression, container):
        # type: (Expression, list[Expression]) -> None
        """Check if a value is in a list of possible values."""
        self.expression = expression
        self.container = container

    def is_literal(self):
        """Check if a set contains only literal values."""
        return all(isinstance(v, Literal) for v in self.container)

    def is_dynamic(self):
        """Check if a set contains only dynamic values."""
        return all(not isinstance(v, Literal) for v in self.container)

    def get_literals(self):
        """Get the values in the set."""
        values = OrderedDict()

        for literal in self.container:  # type: Literal
            if not isinstance(literal, Literal):
                continue
            k = literal.value
            if isinstance(literal, String):
                values.setdefault(fold_case(k), literal)
            else:
                values[k] = literal
        return values

    def __and__(self, other):
        """Perform an intersection between two sets for boolean AND."""
        if isinstance(other, InSet) and self.expression == other.expression:
            if self.is_literal() and other.is_literal():
                container1 = self.get_literals()
                container2 = other.get_literals()

                reduced = [v for k, v in container1.items() if k in container2]
                return InSet(self.expression, reduced).optimize()

        elif isinstance(other, Not):
            if isinstance(other.term, InSet) and self.expression == other.term.expression:
                # Check if one set is being subtracted from another
                if self.is_literal() and other.term.is_literal():
                    container1 = self.get_literals()
                    container2 = other.term.get_literals()

                    reduced = [v for k, v in container1.items() if k not in container2]
                    return InSet(self.expression, reduced).optimize()

        elif isinstance(other, Comparison) and other.comparator == Comparison.EQ and self.expression == other.left:
            if self.is_literal() and isinstance(other.right, Literal):
                return super(InSet, self).__and__(InSet(other.left, [other.right])).optimize()

        elif isinstance(other, Comparison) and other.comparator == Comparison.NE and self.expression == other.left:
            if self.is_literal() and isinstance(other.right, Literal):
                return super(InSet, self).__and__(~ InSet(other.left, [other.right])).optimize()

        return super(InSet, self).__and__(other)

    def __or__(self, other):
        """Perform a union between two sets for boolean OR."""
        if isinstance(other, InSet) and self.expression == other.expression:
            if self.is_literal() and other.is_literal():
                container = self.get_literals()
                for k, v in other.get_literals().items():
                    container.setdefault(k, v)

                union = [v for v in container.values()]
                return InSet(self.expression, union).optimize()

        elif isinstance(other, Comparison) and self.expression == other.left:
            if self.is_literal() and isinstance(other.right, Literal):
                return self.__or__(InSet(other.left, [other.right]))

        return super(InSet, self).__or__(other)

    def split_literals(self):
        """Split the set lookup into static values and dynamic values."""
        if self.is_dynamic() or self.is_literal():
            return self

        literals = InSet(self.expression, [])
        dynamic = InSet(self.expression, [])
        for item in self.container:
            if isinstance(item, Literal):
                literals.container.append(item)
            else:
                dynamic.container.append(item)

        return literals.optimize() | dynamic.optimize()

    @property
    def synonym(self):
        """Get an equivalent node that does performs multiple comparisons with 'or' and '=='."""
        return Or([Comparison(self.expression, Comparison.EQ, v) for v in self.container])

    def _render(self, negate=False):
        values = [v.render() for v in self.container]
        expr = self.expression.render(self.precedence)
        operator = 'not in' if negate else 'in'

        if len(self.container) > 3 and sum(len(v) for v in values) > 40:
            delim = ',\n'
            return '{lhs} {op} (\n{rhs}\n)'.format(lhs=expr, op=operator, rhs=self.indent(delim.join(values)))
        else:
            delim = ', '
            return '{lhs} {op} ({rhs})'.format(lhs=expr, op=operator, rhs=delim.join(values))


class BaseCompound(Expression):
    """Combine multiple expressions with a single operator."""

    __slots__ = 'terms',
    operator = None  # type: str

    def __init__(self, terms):
        """Combine multiple expressions with an operator.

        :param list[Expression] terms: List of terms
        """
        self.terms = terms

    def _render(self):
        scoped_terms = [term.render(self.precedence) for term in self.terms]
        if len(scoped_terms) == 1:
            return scoped_terms[0]

        if len(self.terms) > 4 or any(isinstance(t, (BaseCompound, NamedSubquery, InSet)) for t in self.terms):
            delim = ' {}\n'.format(self.operator)
            indented = [self.indent(t) for t in scoped_terms]
            return delim.join(indented).lstrip()
        else:
            delim = ' {} '.format(self.operator)
            return delim.join(scoped_terms).lstrip()


class Not(Expression):
    """Negate a boolean expression."""

    __slots__ = 'term',
    precedence = Comparison.precedence + 1
    template = Template('not $term')

    def __init__(self, term):
        """Init.

        :param Expression term: The query node to negate
        """
        self.term = term

    def demorgans(self):
        """Apply DeMorgan's law."""
        if isinstance(self.term, Or):
            return And([(~ t).optimize() for t in self.term.terms]).optimize()

        elif isinstance(self.term, And):
            return Or([(~ t).optimize() for t in self.term.terms]).optimize()

        else:
            return ~ self.term.optimize()

    def __invert__(self):
        """Convert ``not not X`` to X."""
        return self.term.optimize()

    def render(self, precedence=None):
        """Convert wildcard functions back to the short hand syntax."""
        if isinstance(self.term, InSet):
            return self.term.render(precedence, negate=True)

        if isinstance(self.term, FunctionCall) and self.term.name == 'wildcard':
            if len(self.term.arguments) == 2 and isinstance(self.term.arguments[1], String):
                lhs, rhs = self.term.arguments
                return Comparison(lhs, Comparison.NE, rhs).render(precedence)
        return super(Not, self).render(precedence)


class And(BaseCompound):
    """Perform a boolean ``and`` on a list of expressions."""

    precedence = Not.precedence + 1
    operator = 'and'


class Or(BaseCompound):
    """Perform a boolean ``or`` on a list of expressions."""

    precedence = And.precedence + 1
    operator = 'or'


class EventQuery(EqlNode):
    """Query over a specific event type with a boolean condition."""

    __slots__ = 'event_type', 'query'
    template = Template('$event_type where $query')

    def __init__(self, event_type, query):
        """Init.

        :param str event_type: One of the event types in the repo sensor/eventing_schema
        :param query: The query scoped to the event type
        """
        self.event_type = event_type
        self.query = query

    def _render(self):
        query_text = self.query.render()
        if '\n' in query_text:
            return '{} where\n{}'.format(self.event_type, self.indent(query_text))

        return super(EventQuery, self)._render()


class NamedParams(EqlNode):
    """An EQL node for key-value named parameters."""

    __slots__ = 'kv',

    def __init__(self, kv=None):
        """Key value store for EQL parameters.

        :param dict[str, Expression] kv: The named key-value parameters.
        """
        self.kv = kv or {}

    def _render(self):
        return ' '.join('{}={}'.format(k, v.render(Literal.precedence)) for k, v in self.kv.items())


class SubqueryBy(EqlNode):
    """Node for holding the :class:`~EventQuery` and parameters to join on."""

    __slots__ = 'query', 'join_values', 'fork',

    def __init__(self, query, join_values=None, fork=None):
        """Init.

        :param EventQuery query: The event query enclosed in the term
        :param list[Expression] join_values: The field to join values on
        :param bool fork: Toggle for copying instead of moving a sequence on match
        """
        self.query = query
        self.join_values = join_values or []
        self.fork = fork

    @property
    def params(self):
        """Keep params for backwards compatibility."""
        params = {}
        if self.fork is not None:
            params["fork"] = Boolean(self.fork)
        return NamedParams(params)

    def _render(self):
        text = "[{}]".format(self.query.render())
        params = self.params.render()
        if len(params):
            text += ' ' + params

        if len(self.join_values):
            text += ' by {}'.format(', '.join(jv.render() for jv in self.join_values))
        return text


class Join(EqlNode):
    """Another boolean query that can join multiple events that share common values."""

    __slots__ = 'queries', 'close'

    def __init__(self, queries, close=None):
        """Init.

        :param list[SubqueryBy] queries:
        :param SubqueryBy close: The condition to purge all join state.
        """
        self.queries = queries
        self.close = close

    def _render(self):
        text = 'join\n'
        text += self.indent('\n'.join(query.render() for query in self.queries))

        if self.close:
            text += '\nuntil\n' + self.indent(self.close.render())
        return text


class Sequence(EqlNode):
    """Sequence is very similar to join, but enforces an ordering.

    Sequence supports the ``until`` keyword, which indicates an event that causes it to terminate early.
    """

    __slots__ = 'queries', 'max_span', 'close'

    def __init__(self, queries, max_span=None, close=None):
        """Create a Sequence of multiple events.

        :param list[SubqueryBy] queries: List of queries to be sequenced
        :param TimeUnit max_span: Dictionary of timing parameters for the sequence.
        :param SubqueryBy close: An optional query that causes all sequence state to expire
        """
        self.queries = queries
        self.max_span = max_span
        self.close = close

    @property
    def params(self):
        """Keep params for backwards compatibility."""
        params = {}
        if self.max_span is not None:
            params["maxspan"] = self.max_span
        return NamedParams(params)

    def _render(self):
        text = 'sequence'
        if self.max_span:
            text += ' with maxspan={}'.format(self.max_span.render())
        text += '\n'
        text += self.indent('\n'.join(query.render() for query in self.queries))

        if self.close:
            text += '\nuntil\n' + self.indent(self.close.render())
        return text


# noinspection PyAbstractClass
class PipeCommand(EqlNode, SignatureMixin):
    """Base class for an EQL pipe."""

    __slots__ = 'arguments',
    name = None  # type: str
    lookup = {}  # type: dict[str, PipeCommand|type]

    def __init__(self, arguments=None):  # type: (list[Expression]) -> None
        """Create a pipe with optional arguments."""
        self.arguments = arguments or []
        super(PipeCommand, self).__init__()

    @classmethod
    def register(cls, name):
        """Register a pipe class by name."""
        def decorator(pipe_class):
            pipe_class.name = name
            if name in cls.lookup:
                raise KeyError("Pipe {} already registered as {}".format(cls.lookup[name], name))
            cls.lookup[name] = pipe_class
            return pipe_class
        return decorator

    @classmethod
    def output_schemas(cls, arguments, event_schemas):
        # type: (list[NodeInfo], list[Schema]) -> list[Schema]
        """Output a list of schemas for each event in the pipe."""
        return event_schemas

    def _render(self):
        if len(self.arguments) == 0:
            return self.name
        return self.name + ' ' + ', '.join(arg.render() for arg in self.arguments)


class PipedQuery(EqlNode):
    """List of all the pipes."""

    __slots__ = 'first', 'pipes'

    def __init__(self, first, pipes=None):
        """Init.

        :param EventQuery|Join|Sequence first: first query
        :param list[PipeCommand] pipes: List of all of the following pipes
        """
        self.first = first
        self.pipes = pipes or []

    def _render(self):
        all_pipes = [self.first] + self.pipes
        return '\n| '.join(pipe.render() for pipe in all_pipes)


class EqlAnalytic(EqlNode):
    """Analytics are the top-level nodes for matching and returning events."""

    __slots__ = 'query', 'metadata'

    def __init__(self, query, metadata=None):
        """Init.

        :param PipedQuery query: Analytic query
        :param dict metadata: Metadata for the analytic
        """
        self.query = query
        self.metadata = metadata or {}

    @property
    def id(self):
        """Return the ID from metadata."""
        return self.metadata.get('id')

    @property
    def name(self):
        """Return the name from metadata."""
        return self.metadata.get('name')

    def __unicode__(self):
        """Print a string instead of the dictionary that render returns."""
        return self.query.__unicode__()

    def __str__(self):
        """Print a string instead of the dictionary that render returns."""
        return self.query.__str__()

    def _render(self):
        return {'metadata': self.metadata, 'query': self.query.render()}


class Definition(object):
    """EQL definitions used for pre-processor expansion."""

    __slots__ = 'name',

    def __init__(self, name):
        """Create a generic definition with a name.

        :param str name: The name of the macro
        """
        self.name = name
        super(Definition, self).__init__()


class Constant(Definition, EqlNode):
    """EQL constant which binds a literal to a name."""

    __slots__ = 'name', 'value',
    template = Template('const $name = $value')

    def __init__(self, name, value):  # type: (str, Literal) -> None
        """Create an EQL literal constant."""
        super(Constant, self).__init__(name)
        self.value = value


class BaseMacro(Definition):
    """Base macro class."""

    def expand(self, arguments):
        """Expand a macro with a set of arguments."""
        raise NotImplementedError


class CustomMacro(BaseMacro):
    """Custom macro class to use Python callbacks to transform trees."""

    def __init__(self, name, callback):
        """Python macro to allow for more dynamic or sophisticated macros.

        :param str name: The name of the macro.
        :param (list[EqlNode]) -> EqlNode callback: A callback to expand out the macro.
        """
        super(CustomMacro, self).__init__(name)
        self.callback = callback

    def expand(self, arguments):
        """Make the callback do the dirty work for expanding the AST."""
        node = self.callback(arguments)
        return node.optimize(recursive=True)

    @classmethod
    def from_name(cls, name):
        """Decorator to convert a function into a :class:`~CustomMacro` object."""
        def decorator(f):
            return CustomMacro(name, f)
        return decorator


class Macro(BaseMacro, EqlNode):
    """Class for a macro on a node, to allow for client-side expansion."""

    __slots__ = 'name', 'parameters', 'expression'
    template = Template('macro $name($parameters) $expression')
    delims = {'parameters': ', '}

    def __init__(self, name, parameters, expression):
        """Create a named macro that takes a list of arguments and returns a paramaterized expression.

        :param str name: The name of the macro.
        :param list[str]: The names of the parameters.
        :param Expression expression: The parameterized expression to return.
        """
        BaseMacro.__init__(self, name)
        EqlNode.__init__(self)
        self.parameters = parameters
        self.expression = expression

    def expand(self, arguments):
        """Expand a node.

        :param list[BaseNode node] arguments: The arguments the macro is called with
        :param Walker walker: An optional syntax tree walker.
        :param bool optimize: Return an optimized copy of the AST
        :rtype: BaseNode
        """
        if len(arguments) != len(self.parameters):
            raise ValueError("Macro {} expected {} arguments but received {}".format(
                self.name, len(self.parameters), len(arguments)))

        lookup = dict(zip(self.parameters, arguments))

        def _walk_field(node):
            # type: (Field) -> Field
            if node.base in lookup:
                argument_value = lookup[node.base]
                if node.path:
                    if not isinstance(argument_value, Field):
                        raise EqlCompileError("Invalid expansion: {} ({}={})".format(node, node.base, argument_value))

                    # extend the attributes being retrieved
                    return Field(argument_value.base, list(argument_value.path) + list(node.path))

                return argument_value

            return node

        walker = RecursiveWalker()

        walker.register_func(Field, _walk_field)
        expanded = walker.walk(self.expression)
        return expanded.optimize(recursive=True)

    def _render(self):
        expr = self.expression.render()
        if '\n' in expr or len(expr) > 40:
            expr = '\n' + self.indent(expr)
            return self.template.substitute(name=self.name, parameters=', '.join(self.parameters), expression=expr)
        return super(Macro, self)._render()


class PreProcessor(ParserConfig):
    """An EQL preprocessor stores definitions and is used for macro expansion and constants."""

    def __init__(self, definitions=None):
        """Initialize a preprocessor environment that can load definitions."""
        self.constants = OrderedDict()  # type: dict[str, Constant]
        self.macros = OrderedDict()  # type: dict[str, BaseMacro|CustomMacro|Maco]

        class PreProcessorWalker(RecursiveWalker):
            """Custom walker class for this preprocessor."""

            preprocessor = self

            def _walk_field(self, node, *args, **kwargs):
                if node.base in self.preprocessor.constants and not node.path:
                    return self.preprocessor.constants[node.base].value
                return self._walk_base_node(node, *args, **kwargs)

            def _walk_function_call(self, node, *args, **kwargs):
                if node.name in self.preprocessor.macros:
                    macro = self.preprocessor.macros[node.name]
                    arguments = [self.walk(arg, *args, **kwargs) for arg in node.arguments]
                    return macro.expand(arguments)
                return self._walk_base_node(node, *args, **kwargs)

        self.walker_cls = PreProcessorWalker
        ParserConfig.__init__(self, preprocessor=self)
        self.add_definitions(definitions or [])

    def add_definitions(self, definitions):
        """Add a list of definitions."""
        for definition in definitions:
            self.add_definition(definition)

    def add_definition(self, definition):  # type: (BaseMacro|Constant) -> None
        """Add a named definition to the preprocessor."""
        name = definition.name
        if isinstance(definition, BaseMacro):
            # The macro may call into other macros so it should be expanded
            expanded_macro = self.expand(definition)
            self.macros[name] = expanded_macro
        elif isinstance(definition, Constant):
            if name in self.constants:
                raise KeyError("Constant {} already defined".format(name))
            self.constants[name] = definition

    def expand(self, root):
        """Expand the function calls that match registered macros.

        :param EqlNode root: The input node, macro, expression, etc.
        :param bool optimize: Toggle AST optimizations while expanding
        :rtype: EqlNode
        """
        if not self.constants and not self.macros:
            return root

        return self.walker_cls().walk(root)

    def copy(self):
        """Create a shallow copy of a preprocessor."""
        preprocessor = PreProcessor()
        preprocessor.constants.update(self.constants)
        preprocessor.macros.update(self.macros)
        return preprocessor


# circular dependency
from .walkers import Walker, RecursiveWalker  # noqa: E402
from .functions import get_function  # noqa: E402
from .optimizer import Optimizer  # noqa: E402
