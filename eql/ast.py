"""EQL syntax tree nodes/schema."""
from __future__ import unicode_literals

import datetime
import re
from collections import OrderedDict
from operator import lt, le, eq, ne, ge, gt
from string import Template

from eql.utils import to_unicode, is_string, is_number


__all__ = (
    # base classes
    "BaseNode",
    "AstWalker",

    "Expression",
    "EqlNode",

    # Literals
    "Literal",
    "String",
    "Number",
    "Null",
    "Boolean",
    "TimeRange",

    # fields and subfields
    "Field",

    # boolean logic
    "Comparison",
    "InSet",
    "And",
    "Or",
    "Not",
    "FunctionCall",

    # queries
    "EventQuery",
    "NamedSubquery",
    "NamedParams",
    "SubqueryBy",
    "Join",
    "Sequence",

    # pipes
    "PipeCommand",
    "ByPipe",
    "HeadPipe",
    "TailPipe",
    "SortPipe",
    "UniquePipe",
    "CountPipe",
    "FilterPipe",
    "UniqueCountPipe",

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

    def optimize(self):
        """Optimize an AST."""
        return self

    def __eq__(self, other):
        """Check if two ASTs are equivalent."""
        return type(self) == type(other) and list(self.iter_slots()) == list(other.iter_slots())

    def __ne__(self, other):
        """Check if two ASTs are not equivalent."""
        return not self == other

    def render(self, precedence=None):
        """Render the AST in the target language."""
        if not self.template:
            raise NotImplementedError()

        dicted = {}
        for name, value in self.iter_slots():
            if isinstance(value, (list, tuple)):
                delim = self.delims[name]
                value = [v.render(self.precedence) if isinstance(v, BaseNode) else v for v in value]
                value = delim.join(v for v in value)
            elif isinstance(value, BaseNode):
                value = value.render(self.precedence)
            dicted[name] = value
        return self.template.substitute(dicted)

    def __repr__(self):
        """Python representation of the AST."""
        return "{}({})".format(type(self).__name__, ", ".join('{}={}'.format(name, repr(slot))
                                                              for name, slot in self.iter_slots()))

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


class AstWalker(object):
    """Base class that provides functionality for walking abstract syntax trees of eql.BaseNode."""

    @classmethod
    def walk(cls, node, func):
        """Walk the syntax tree top-down, until callback returns False.

        :param BaseNode node: Any AST node
        :param (BaseNode) -> bool func: Walk function
        """
        if isinstance(node, BaseNode):
            if not func(node):
                return

            for slot, child in node.iter_slots():
                cls.walk(child, func)
        elif isinstance(node, (list, tuple)):
            for child in node:
                cls.walk(child, func)
        elif isinstance(node, dict):
            for key, child in node.items():
                cls.walk(child, func)

    def transform(self, node, func, optimize=True):
        """Recursively transform the syntax tree by walking bottom-up.

        :param BaseNode node: Any AST node
        :param function func: Callback function for walking with the signature
            ``func(original_node, transformed_node) -> bool``
        :param bool optimize: Return an optimized copy of the AST
        :rtype: BaseNode
        """
        if isinstance(node, BaseNode):
            cls = type(node)
            args = [self.transform(child, func, optimize=optimize) for _, child in node.iter_slots()]
            transformed = cls(*args)
            if optimize:
                transformed = transformed.optimize()

            output = func(transformed, node)  # type: BaseNode

            if optimize:
                return output.optimize()
            return output
        elif isinstance(node, (list, tuple)):
            return [self.transform(child, func, optimize=optimize) for child in node]
        elif isinstance(node, dict):
            return {key: self.transform(child, func, optimize=optimize) for key, child in node.items()}
        else:
            return node

    def copy(self, node, optimize=True):
        """Create a copy of an AST.

        :param BaseNode node: Any valid AST
        :param bool optimize: Return an optimized copy of the AST
        :rtype: BaseNode
        """
        return self.transform(node, lambda copy, original: copy, optimize=optimize)


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

    def render(self, precedence=None):
        """Render an EQL node and add parentheses to support orders of operation."""
        rendered = self._render()
        if precedence is not None and self.precedence is not None and self.precedence > precedence:
            return '({})'.format(rendered)
        return rendered


# noinspection PyAbstractClass
class Expression(EqlNode):
    """Base class for expressions."""

    precedence = 0

    def __and__(self, other):
        """Boolean AND between two AST nodes."""
        if isinstance(other, Literal):
            if other.value:
                return self
            return Boolean(False)

        if isinstance(other, And):
            return And([self] + other.terms)
        return And([self, other])

    def __or__(self, other):
        """"Boolean OR between two AST nodes."""
        if isinstance(other, Literal):
            if other.value:
                return Boolean(True)

        if isinstance(other, Or):
            return Or([self] + other.terms)
        return Or([self, other])

    def __invert__(self):
        """Negate an expression with Not."""
        return Not(self)


class Literal(Expression):
    """Static value."""

    __slots__ = 'value',
    precedence = Expression.precedence + 1

    def __init__(self, value):
        """Create an EQL value from a python value."""
        assert type(self) is not Literal, "Illegal usage of Literal AST node"
        self.value = value

    def __and__(self, other):
        """Shortcut ANDing of Static Value nodes together."""
        if isinstance(other, Literal):
            return Boolean(self.value and other.value)
        elif self.value:
            return other
        else:
            return Boolean(False)

    def __or__(self, other):
        """Shortcut ORing of Static Value nodes together."""
        if isinstance(other, Literal):
            return Boolean(self.value or other.value)
        elif self.value:
            return self
        else:
            return other

    def __invert__(self):
        """Negate a static value."""
        return Boolean(not self.value)


class Boolean(Literal):
    """Boolean literal."""

    def _render(self):
        return 'true' if self.value else 'false'


class Null(Literal):
    """Null literal."""

    def __init__(self, value=None):
        """Null literal value."""
        super(Null, self).__init__(None)

    def _render(self):
        return 'null'


class Number(Literal):
    """Numeric literal."""

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
            return cls.reverse_patterns.get(sub.group(), sub.group())
        return re.sub(r"\\.", replace_callback, s)

    def _render(self):
        return '"{}"'.format(self.escape(self.value))


class TimeRange(Expression):
    """EQL node for an interval of time."""

    __slots__ = 'delta',
    precedence = Expression.precedence + 1

    def __init__(self, delta):  # type: (datetime.timedelta) -> None
        """EQL time interval."""
        self.delta = delta

    @classmethod
    def convert(cls, node):
        """Convert a StaticValue to a time range."""
        if isinstance(node, TimeRange):
            return node
        elif isinstance(node, Number):
            return TimeRange(datetime.timedelta(seconds=node.value))

    def _render(self):
        interval = self.delta.total_seconds()
        second = 1
        minute = 60 * second
        hour = minute * 60
        day = hour * 24
        decimal = interval
        unit = 's'

        if interval >= day:
            decimal = float(interval) / day
            unit = 'd'
        elif interval >= hour:
            decimal = float(interval) / hour
            unit = 'h'
        elif interval >= minute:
            if interval % minute == 0 or (interval % second) != 0:
                decimal = float(interval) / minute
                unit = 'm'

        # Drop fractional part if it's 0
        if decimal == int(decimal):
            decimal = int(decimal)
        return '{}{}'.format(decimal, unit)


class Field(Expression):
    """Variables and paths in scope of the event."""

    EVENTS = 'events'

    __slots__ = 'base', 'path',
    precedence = Expression.precedence + 1

    def __init__(self, base, path=None):
        """Query the event for the field expression.

        :param str base: The root field
        :param list[str|int] path: The sub fields and array positions
        """
        self.base = base
        self.path = path or []

    def query_multiple_events(self):  # type: () -> (int, Field)
        """Get the index into the event array and query."""
        if self.base == Field.EVENTS and len(self.path) >= 2:
            if is_number(self.path[0]) and is_string(self.path[1]):
                return self.path[0], Field(self.path[1], self.path[2:])
        return 0, self

    def _render(self):
        text = self.base
        for key in self.path:
            if is_number(key):
                text += "[{}]".format(key)
            else:
                text += ".{}".format(key)
        return text


class FunctionCall(Expression):
    """A call into a user-defined function by name and a list of arguments."""

    __slots__ = 'name', 'arguments'
    precedence = Literal.precedence + 1
    template = Template('$name($arguments)')
    delims = {'arguments': ', '}

    def __init__(self, name, arguments):
        """Call the function by name.

        :param str name: The name of the user-defined function
        :param list[Expression] arguments: Arguments to pass into the function.
        """
        self.name = name
        self.arguments = arguments or []

    def optimize(self):
        """Optimize function calls that can be determined at compile time."""
        if self.name == 'wildcard':
            if any(isinstance(arg, Literal) and not isinstance(arg, String) for arg in self.arguments):
                return Boolean(False)

            if len(self.arguments) >= 2 and all(isinstance(arg, String) for arg in self.arguments):
                source = self.arguments[0].value
                regex = '|'.join('^{}$'.format(r'.*?'.join(re.escape(sequence)
                                                           for sequence in literal.value.split('*')))
                                 for literal in self.arguments[1:])
                return Boolean(re.match(regex, source, re.IGNORECASE) is not None)
        elif self.name == 'length' and all(isinstance(arg, String) for arg in self.arguments):
            return Number(len(*(arg.value for arg in self.arguments)))
        return self

    def render(self, precedence=None):
        """Convert wildcards back to the short hand syntax."""
        if self.name == 'wildcard' and len(self.arguments) == 2 and isinstance(self.arguments[1], String):
            lhs, rhs = self.arguments
            return Comparison(lhs, Comparison.EQ, rhs).render(precedence)

        return super(FunctionCall, self).render()


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


class Comparison(Expression):
    """Represents a comparison between two values, as in ``<expr> <comparator> <expr>``.

    Comparison operators include ``==``, ``!=``, ``<``, ``<=``, ``>=``, and ``>``.
    """

    __slots__ = 'left', 'comparator', 'right'
    LT, LE, EQ, NE, GE, GT = ('<', '<=', '==', '!=', '>=', '>')

    func_lookup = {LT: lt, LE: le, EQ: eq, NE: ne, GE: ge, GT: gt}
    precedence = NamedSubquery.precedence + 1
    template = Template('$left $comparator $right')

    def __init__(self, left, comparator, right):
        # type: (Expression, str, Expression) -> None
        """Compare two fields or values to each other."""
        self.left = left
        self.comparator = comparator
        self.right = right
        self.function = self.func_lookup[comparator]

    def optimize(self):
        """Optimize comparisons against literal values."""
        if isinstance(self.left, Literal) and isinstance(self.right, Literal):
            lhs = self.left.value
            rhs = self.right.value

            # Check that the types match first
            if not isinstance(self.right, type(self.left)):
                return Boolean(self.comparator == Comparison.NE)

            if isinstance(self.left, String):
                lhs = lhs.lower()
                rhs = rhs.lower()

            return Boolean(self.function(lhs, rhs))

        # assumes calling the same function twice with the same args returns the same result
        elif self.left == self.right:
            return Boolean(self.comparator in (Comparison.EQ, Comparison.LE, Comparison.GE))

        return self


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

    def _get_literals(self):
        """Get the values in the set."""
        values = OrderedDict()

        for literal in self.container:  # type: Literal
            if not isinstance(literal, Literal):
                continue
            k = literal.value
            if isinstance(literal, String):
                values.setdefault(k.lower(), literal)
            else:
                values[k] = literal
        return values

    def __and__(self, other):
        """Perform an intersection between two sets for boolean OR."""
        if isinstance(other, InSet) and self.expression == other.expression:
            if self.is_literal() and other.is_literal():
                container1 = self._get_literals()
                container2 = other._get_literals()

                intersection = [v for k, v in container1.items() if k in container2]
                return InSet(self.expression, intersection).optimize()

        return super(InSet, self).__and__(other)

    def __or__(self, other):
        """Perform a union between two sets for boolean OR."""
        if isinstance(other, InSet) and self.expression == other.expression:
            if self.is_literal() and other.is_literal():
                container = self._get_literals()
                for k, v in other._get_literals().items():
                    container.setdefault(k, v)

                union = [v for v in container.values()]
                return InSet(self.expression, union).optimize()

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

    def optimize(self):
        """Optimize the AST."""
        expression = self.expression

        # move all the literals to the front, preserve their ordering
        literals = [v for k, v in self._get_literals().items()]
        dynamic = [v for v in self.container if not isinstance(v, Literal)]
        container = literals + dynamic

        # check to see if a literal value is in the list of literal values
        if isinstance(self.expression, Literal):
            value = self.expression.value
            if is_string(value):
                value = value.lower()
            if value in self._get_literals():
                return Boolean(True)
            container = dynamic

        if len(container) == 0:
            return Boolean(False)
        elif len(container) == 1:
            return Comparison(expression, Comparison.EQ, container[0]).optimize()
        elif expression in container:
            return Boolean(True)

        return InSet(expression, container)

    @property
    def synonym(self):
        """Get an equivalent node that does performs multiple comparisons with 'or' and '=='."""
        return Or([Comparison(self.expression, Comparison.EQ, v) for v in self.container])

    def _render(self):
        values = [v.render() for v in self.container]
        expr = self.expression.render(self.precedence)

        if len(self.container) > 3 and sum(len(v) for v in values) > 40:
            delim = ',\n'
            return '{} in (\n{}\n)'.format(expr, self.indent(delim.join(values)))
        else:
            delim = ', '
            return '{} in ({})'.format(expr, delim.join(values))


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

    def optimize(self):
        """Optimize NOT terms, by flattening them."""
        return ~ self.term.optimize()

    def __invert__(self):
        """Convert ``not not X`` to X."""
        return self.term

    def render(self, precedence=None):
        """Convert wildcard functions back to the short hand syntax."""
        if isinstance(self.term, FunctionCall) and self.term.name == 'wildcard':
            if len(self.term.arguments) == 2 and isinstance(self.term.arguments[1], String):
                lhs, rhs = self.term.arguments
                return Comparison(lhs, Comparison.NE, rhs).render(precedence)
        return super(Not, self).render()


class And(BaseCompound):
    """Perform a boolean ``and`` on a list of expressions."""

    precedence = Not.precedence + 1
    operator = 'and'

    def optimize(self):
        """Optimize AND terms, by flattening them."""
        node = self.terms[0]
        for term in self.terms[1:]:
            node &= term
        return node

    def __and__(self, other):
        """Flatten multiple ``and`` terms."""
        terms = self.terms
        if isinstance(other, And):
            terms.extend(other.terms)
        else:
            terms.append(other)
        return And(terms)


class Or(BaseCompound):
    """Perform a boolean ``or`` on a list of expressions."""

    precedence = And.precedence + 1
    operator = 'or'

    def optimize(self):
        """Optimize OR terms, by flattening them."""
        node = self.terms[0]
        for term in self.terms[1:]:
            node |= term
        return node

    def __or__(self, other):
        """Flatten multiple ``or`` terms."""
        terms = self.terms
        if isinstance(other, Or):
            terms.extend(other.terms)
        else:
            terms.append(other)
        return Or(terms)


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

    __slots__ = 'query', 'params', 'join_values'

    def __init__(self, query, params=None, join_values=None):
        """Init.

        :param EventQuery query: The event query enclosed in the term
        :param NamedParams params: The parameters for the query.
        :param list[Expression] join_values: The field to join values on
        """
        self.query = query
        self.params = params or NamedParams()
        self.join_values = join_values or []

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

    __slots__ = 'queries', 'params', 'close'

    def __init__(self, queries, params=None, close=None):
        """Create a Sequence of multiple events.

        :param list[SubqueryBy] queries: List of queries to be sequenced
        :param NamedParams params: Dictionary of timing parameters for the sequence.
        :param SubqueryBy close: An optional query that causes all sequence state to expire
        """
        self.queries = queries
        self.params = params or NamedParams()
        self.close = close

    def _render(self):
        text = 'sequence'
        params = self.params.render()
        if params:
            text += ' with {}'.format(self.params.render())
        text += '\n'
        text += self.indent('\n'.join(query.render() for query in self.queries))

        if self.close:
            text += '\nuntil\n' + self.indent(self.close.render())
        return text


# noinspection PyAbstractClass
class PipeCommand(EqlNode):
    """Base class for an EQL pipe."""

    __slots__ = 'arguments',
    pipe_name = None  # type: str
    lookup = {}  # type: dict[str, PipeCommand|type]
    minimum_args = None
    maximum_args = None

    def __init__(self, arguments=None):  # type: (list[Expression]) -> None
        """Create a pipe with optional arguments."""
        self.arguments = arguments or []
        super(PipeCommand, self).__init__()

    def validate(self):
        """Find the first invalid argument. Return None if all are valid."""
        pass

    @classmethod
    def register(cls, name):
        """Register a pipe class by name."""
        def decorator(pipe_class):
            pipe_class.pipe_name = name
            if name in cls.lookup:
                raise KeyError("Pipe {} already registered as {}".format(cls.lookup[name], name))
            cls.lookup[name] = pipe_class
            return pipe_class
        return decorator

    def _render(self):
        if len(self.arguments) == 0:
            return self.pipe_name
        return self.pipe_name + ' ' + ', '.join(arg.render() for arg in self.arguments)


class ByPipe(PipeCommand):
    """Pipe that takes a value (field, function, etc.) as a key."""

    minimum_args = 1

    def validate(self):
        """Find the first invalid argument. Return None if all are valid."""
        for i, arg in enumerate(self.arguments):
            if isinstance(arg, Literal) or isinstance(arg, NamedSubquery):
                return i


@PipeCommand.register('head')
class HeadPipe(PipeCommand):
    """Node representing the head pipe, analogous to the unix head command."""

    maximum_args = 1
    DEFAULT = 50

    @property
    def count(self):  # type: () -> int
        """Get the number of elements to emit."""
        if len(self.arguments) == 0:
            return self.DEFAULT
        return self.arguments[0].value

    def validate(self):
        """Find the first invalid argument. Return None if all are valid."""
        if len(self.arguments) > 0:
            arg = self.arguments[0]
            if not (isinstance(arg, Literal) and isinstance(arg.value, int) and arg.value > 0):
                return 0


@PipeCommand.register('tail')
class TailPipe(PipeCommand):
    """Node representing the tail pipe, analogous to the unix tail command."""

    maximum_args = 1
    DEFAULT = 50

    @property
    def count(self):  # type: () -> int
        """Get the number of elements to emit."""
        if len(self.arguments) == 0:
            return self.DEFAULT
        return self.arguments[0].value

    def validate(self):
        """Find the first invalid argument. Return None if all are valid."""
        if len(self.arguments) == 0:
            return
        elif len(self.arguments) > 1:
            return 1
        else:
            arg = self.arguments[0]
            if not (isinstance(arg, Literal) and isinstance(arg.value, int)) or arg.value <= 0:
                return 0


@PipeCommand.register('sort')
class SortPipe(ByPipe):
    """Sorts the pipes by field comparisons."""


@PipeCommand.register('unique')
class UniquePipe(ByPipe):
    """Filters events on a per-field basis, and only outputs the first event seen for a field."""


@PipeCommand.register('count')
class CountPipe(ByPipe):
    """Counts number of events that match a field, or total number of events if none specified."""

    minimum_args = 0


@PipeCommand.register('filter')
class FilterPipe(PipeCommand):
    """Takes data coming into an existing pipe and filters it further."""

    minimum_args = 1
    maximum_args = 1

    @property
    def expression(self):
        """Get the filter expression."""
        return self.arguments[0]

    def validate(self):
        """Validate that exactly one expression is sent."""
        if not isinstance(self.expression, Expression):
            return 0


@PipeCommand.register('unique_count')
@PipeCommand.register('ucount')
class UniqueCountPipe(ByPipe):
    """Returns unique results but adds a count field."""


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

    __slots__ = 'query', 'actions', 'metadata'

    def __init__(self, query, actions=None, metadata=None):
        """Init.

        :param PipedQuery query: Analytic query
        :param list[str] actions: List of actions for the query
        :param dict metadata: Metadata for the analytic
        """
        self.query = query
        self.actions = actions
        self.metadata = metadata or {}

    @property
    def id(self):
        """Return the ID from metadata."""
        return self.metadata.get('id')

    @property
    def name(self):
        """Return the name from metadata."""
        return self.metadata.get('name')

    def __str__(self):
        """Print a string instead of the dictionary that render returns."""
        return self.__repr__()

    def __unicode__(self):
        """Print a string instead of the dictionary that render returns."""
        return self.__repr__()

    def _render(self):
        return {'metadata': self.metadata, 'actions': self.actions, 'query': self.query.render()}


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

    def expand(self, arguments, walker=None, optimize=True):
        """Expand a macro with a set of arguments."""
        raise NotImplementedError


class CustomMacro(BaseMacro):
    """Custom macro class to use Python callbacks to transform trees."""

    def __init__(self, name, callback):
        """Python macro to allow for more dynamic or sophisticated macros.

        :param str name: The name of the macro.
        :param (list[EqlNode], AstWalker) -> EqlNode callback: A callback to expand out the macro.
        """
        super(CustomMacro, self).__init__(name)
        self.callback = callback

    def expand(self, arguments, walker=None, optimize=True):
        """Make the callback do the dirty work for expanding the AST."""
        node = self.callback(arguments, walker)
        if optimize:
            return node.optimize()
        return node

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
        super(Macro, self).__init__(name)
        self.parameters = parameters
        self.expression = expression

    def expand(self, arguments, walker=None, optimize=True):
        """Expand a node.

        :param list[BaseNode node] arguments: The arguments the macro is called with
        :param AstWalker walker: An optional syntax tree walker.
        :param bool optimize: Return an optimized copy of the AST
        :rtype: BaseNode
        """
        if len(arguments) != len(self.parameters):
            raise ValueError("Macro {} expected {} arguments but received {}".format(
                self.name, len(self.parameters), len(arguments)))

        lookup = dict(zip(self.parameters, arguments))
        walker = walker or AstWalker()

        def expand_variables(node, _):
            """Callback for walking the AST that expands the variables into the passed in expression."""
            if isinstance(node, Field):
                if node.base in lookup and not node.path:
                    node = walker.copy(lookup[node.base])
            return node

        expanded = walker.transform(self.expression, expand_variables, optimize=optimize)
        return expanded

    def _render(self):
        expr = self.expression.render()
        if '\n' in expr or len(expr) > 40:
            expr = '\n' + self.indent(expr)
            return self.template.substitute(name=self.name, parameters=', '.join(self.parameters), expression=expr)
        return super(Macro, self)._render()


class PreProcessor(object):
    """An EQL preprocessor stores definitions and is used for macro expansion and constants."""

    def __init__(self, definitions=None):
        """Initialize a preprocessor environment that can load definitions."""
        self.walker = AstWalker()
        self.constants = OrderedDict()  # type: dict[str, Constant]
        self.macros = OrderedDict()  # type: dict[str, BaseMacro]
        self.add_definitions(definitions or [])

    def add_definitions(self, definitions):
        """Add a list of definitions."""
        for definition in definitions:
            self.add_definition(definition)

    def add_definition(self, definition):  # type: (Definition) -> None
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

    def expand(self, root, optimize=True):
        """Expand the function calls that match registered macros.

        :param EqlNode root: The input node, macro, expression, etc.
        :param bool optimize: Toggle AST optimizations while expanding
        :rtype: EqlNode
        """
        if not optimize and not self.constants and not self.macros:
            return root

        def expand_callback(node, _):
            if isinstance(node, FunctionCall):
                if node.name in self.macros:
                    macro = self.macros[node.name]
                    expanded = macro.expand(node.arguments, self.walker, optimize=optimize)
                    node = expanded
            elif isinstance(node, Field) and not node.path:
                if node.base in self.constants:
                    node = self.constants[node.base].value
            return node
        return self.walker.transform(root, expand_callback, optimize=optimize)

    def copy(self):
        """Create a shallow copy of a preprocessor."""
        preprocessor = PreProcessor()
        preprocessor.constants.update(self.constants)
        preprocessor.macros.update(self.macros)
        return preprocessor
