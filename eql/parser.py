"""Python parser functions for EQL syntax."""
from __future__ import unicode_literals

import datetime
from collections import OrderedDict

import tatsu
import tatsu.exceptions
import tatsu.objectmodel
import tatsu.semantics
import tatsu.walkers

from eql.ast import *  # noqa: F401
from eql.errors import ParseError, SchemaError
from eql.etc import get_etc_file
from eql.schema import EVENT_TYPE_ANY, check_event_name
from eql.utils import is_string, to_unicode


__all__ = (
    "get_preprocessor",
    "parse_definition",
    "parse_definitions",
    "parse_expression",
    "parse_query",
    "parse_analytic",
    "parse_analytics",
)


# Used for time units
SECOND = 1
MINUTE = 60 * SECOND
HOUR = MINUTE * 60
DAY = HOUR * 24

# Time units (shorthand)
units = {
    'second': SECOND,
    'minute': MINUTE,
    'hour': HOUR,
    'hr': HOUR,
    'day': DAY
}


GRAMMAR = None
tatsu_parser = None


class EqlWalker(tatsu.walkers.NodeWalker):
    """Walker of Tatsu semantic model to convert it into a EQL AST."""

    def __init__(self, implied_base=False, implied_any=False, subqueries=True, pipes=True, preprocessor=None):
        """Walker for building an EQL syntax tree from a Tatsu syntax tree.

        :param bool implied_any: Allow for event queries to skip event type and WHERE, replace with 'any where ...'
        :param bool implied_base: Allow for queries to be built with only pipes. Base query becomes 'any where true'
        :param bool subqueries: Toggle support for subqueries (sequence, join, named of, etc.)
        :param bool pipes: Toggle support for pipes
        :param PreProcessor preprocessor: Use an EQL preprocessor to expand definitions and constants while parsing
        """
        super(EqlWalker, self).__init__()
        self.implied_base = implied_base
        self.implied_any = implied_any
        self.preprocessor = preprocessor or PreProcessor()
        self._eql_walker = AstWalker()
        self._subqueries_enabled = subqueries
        self._pipes_enabled = pipes

    @staticmethod
    def _error(node, message, end=False, cls=ParseError):
        """Callback function to walk the AST."""
        params = dict(node.ast)
        for k, value in params.items():
            if isinstance(value, list):
                params[k] = ', '.join([v.render() if isinstance(v, EqlNode) else to_unicode(v) for v in value])
        message = message.format(**params)
        lines = node.parseinfo.text_lines()
        line_number = node.parseinfo.line
        if line_number >= len(lines):
            line_number = len(lines) - 1
        bad_line = lines[line_number].rstrip()
        pos = node.parseinfo.endpos if end else node.parseinfo.pos
        return cls(message, line_number, pos, bad_line)

    def _walk_default(self, node, *args, **kwargs):
        """Callback function to walk the AST."""
        if isinstance(node, list):
            return [self.walk(n, *args, **kwargs) for n in node]
        return node

    def walk(self, node, *args, **kwargs):
        """Optimize the AST while walking it."""
        output = super(EqlWalker, self).walk(node, *args, **kwargs)
        if isinstance(output, EqlNode):
            return output.optimize()
        return output

    # literals
    def walk__literal(self, node):
        """Callback function to walk the AST."""
        literal = self.walk(node.value)
        if literal is None:
            return literal
        elif is_string(literal):
            # If a 'raw' string is detected, then only unescape the quote character
            if node.text.startswith('?'):
                quote_char = node.text[-1]
                literal = literal.replace("\\" + quote_char, quote_char)
            else:
                literal = String.unescape(literal)
            return String(to_unicode(literal))
        elif isinstance(literal, bool):
            return Boolean(literal)
        else:
            return Number(literal)

    def walk__time_range(self, node):
        """Callback function to walk the AST."""
        val = self.walk(node.val)
        unit = self.walk(node.unit)
        for name, interval in units.items():
            if name.startswith(unit.rstrip('s') or 's'):
                return TimeRange(datetime.timedelta(seconds=val * interval))

        raise self._error(node.unit, "Unknown time unit")

    # fields
    def walk__field(self, node):
        """Callback function to walk the AST."""
        reserved = 'true', 'false', 'null'
        if node.base in reserved:
            if len(node.sub_fields) != 0:
                raise self._error(node, "Invalid field name {base}")
            elif node.base == 'true':
                return Boolean(True)
            elif node.base == 'false':
                return Boolean(False)
            elif node.base == 'null':
                return Null()
            else:
                raise self._error(node.base, "Unhandled literal")

        path = []

        for sub_field in self.walk(node.sub_fields):
            if is_string(sub_field) and sub_field in reserved:
                raise self._error(node, "Invalid attribute {}".format(sub_field))
            path.append(sub_field)

        if not path and node.base in self.preprocessor.constants:
            constant = self.preprocessor.constants[node.base]
            return constant.value

        return Field(node.base, path)

    # comparisons
    def walk__equals(self, node):
        """Callback function to walk the AST."""
        # May be double or single equals
        return '=='

    def walk__comparison(self, node):
        """Callback function to walk the AST."""
        left = self.walk(node.left)
        right = self.walk(node.right)
        op = self.walk(node.op)

        # there is no special comparator for wildcards, just look for * in the string
        if isinstance(right, String) and '*' in right.value:
            if op == Comparison.EQ:
                return FunctionCall('wildcard', [left, right])
            elif op == Comparison.NE:
                return ~ FunctionCall('wildcard', [left, right])

        return Comparison(left, op, right)

    def walk__and_terms(self, node):
        """Callback function to walk the AST."""
        terms = self.walk(node.terms)
        term = And(terms)
        return term

    def walk__or_terms(self, node):
        """Callback function to walk the AST."""
        terms = self.walk(node.terms)
        term = Or(terms)
        return term

    def walk__not_term(self, node):
        """Callback function to walk the AST."""
        term = Not(self.walk(node.t))
        return term

    def walk__in_set(self, node):
        """Callback function to walk the AST."""
        expr = self.walk(node.expr)
        container = self.walk(node.container)  # type: list[Expression]
        return InSet(expr, container)

    def walk__function_call(self, node):
        """Callback function to walk the AST."""
        args = self.walk(node.args) or []

        if node.name in self.preprocessor.macros:
            macro = self.preprocessor.macros[node.name]
            return macro.expand(args, self._eql_walker)

        return FunctionCall(node.name, args)

    # queries
    def walk__event_query(self, node):
        """Callback function to walk the AST."""
        if node.ast.get('event_type') is None:
            event_type = EVENT_TYPE_ANY
            if not self.implied_any:
                raise self._error(node, "Missing event type and 'where' condition")
        else:
            event_type = node.event_type
            if not check_event_name(event_type):
                raise self._error(node, "Invalid event type: {event_type}", cls=SchemaError)

        return EventQuery(event_type, self.walk(node.cond))

    def walk__pipe(self, node):
        """Callback function to walk the AST."""
        if not self._pipes_enabled:
            raise self._error(node, "Pipes not supported")

        pipe_cls = PipeCommand.lookup.get(node.name)
        if pipe_cls is None:
            raise self._error(node, "Unknown pipe '{name}'")

        pipe = pipe_cls(self.walk(node.args))  # type: PipeCommand
        num_args = len(pipe.arguments)

        error_node = node

        # Try to line up the error message withe the argument that went wrong
        # Strings and numbers don't generate tatsu nodes, so its difficult to recover parseinfo

        if pipe.minimum_args is not None and num_args < pipe.minimum_args:
            message = "Expected {} argument(s) to pipe '{}'".format(pipe.minimum_args, node.name)
            raise self._error(error_node, message, end=True)

        elif pipe.maximum_args is not None and num_args > pipe.maximum_args:
            message = "Expected up to {} argument(s) to pipe '{}'".format(pipe.maximum_args, node.name)
            if isinstance(node.args[pipe.maximum_args], tatsu.semantics.Node):
                error_node = node.args[pipe.maximum_args]
            raise self._error(error_node, message)

        bad_index = pipe.validate()
        if bad_index is not None:
            if isinstance(node.args[bad_index], tatsu.semantics.Node):
                error_node = node.args[bad_index]
            raise self._error(error_node, "Invalid arguments to '{}'".format(node.name))
        return pipe

    def walk__piped_query(self, node):
        """Callback function to walk the AST."""
        if node.query is None:
            first = EventQuery(EVENT_TYPE_ANY, Boolean(True))
            if not self.implied_base:
                raise self._error(node, "Missing base query")
        else:
            first = self.walk(node.query)
        return PipedQuery(first, self.walk(node.pipes))

    def walk__named_query(self, node):
        """Callback function to walk the AST."""
        if not self._subqueries_enabled:
            raise self._error(node, "Subqueries not supported")

        if node.name not in NamedSubquery.supported_types:
            options = ', '.join(NamedSubquery.supported_types)
            raise self._error(node, "Unknown subquery '{name}' of. Available options are: " + options)
        return NamedSubquery(node.name, self.walk(node.query))

    def walk__named_params(self, node, get_param=None):
        """Callback function to walk the AST."""
        params = OrderedDict()
        if get_param is None and len(node.params) > 0:
            raise self._error(node, "Unexpected parameters")

        for param in node.params:
            key, value = get_param(param)
            if key in params:
                raise self._error(param, "Repeated parameter '{k}'")
            params[key] = value
        return NamedParams(params)

    def walk__subquery_by(self, node, num_values=None, get_param=None):
        """Callback function to walk the AST."""
        if not self._subqueries_enabled:
            raise self._error(node, "Subqueries not supported")

        if num_values is not None and num_values != len(node.join_values):
            if len(node.join_values) == 0:
                error_node = node.query
                end = True
            else:
                end = False
                error_node = node.join_values[max(num_values, len(node.join_values)) - 1]
            raise self._error(error_node, "Expected {} value(s)".format(num_values), end=end)

        join_values = self.walk(node.join_values)
        params = self.walk(node.params, get_param=get_param)

        query = self.walk(node.query)
        return SubqueryBy(query, params, join_values)

    def walk__join(self, node):
        """Callback function to walk the AST."""
        if not self._subqueries_enabled:
            raise self._error(node, "Subqueries not supported")

        shared = []
        close = None

        if node.ast.get('shared_by'):
            shared = self.walk(node.shared_by)

        # Figure out how many fields are joined by in the first query, and match across all
        first = self.walk(node.queries[0])
        num_values = len(first.join_values)
        queries = [first]  # type: list[SubqueryBy]
        queries.extend(self.walk(node.queries[1:], num_values=num_values))

        for query in queries:
            query.join_values = shared + query.join_values

        if node.ast.get('until'):
            close = self.walk(node.until, num_values=num_values)  # type: SubqueryBy
            close.join_values = shared + close.join_values

        return Join(queries, close)

    def get_sequence_parameter(self, node):
        """Validate that sequence parameters are working."""
        key, value = self.walk([node.k, node.v])
        value = TimeRange.convert(value)

        if key != 'maxspan':
            raise self._error(node, "Unknown sequence parameter '{}'".format(key))

        if not TimeRange.convert(value) or value.delta < datetime.timedelta(0):
            error_node = node.v if isinstance(node.v, tatsu.objectmodel.Node) else node
            raise self._error(error_node, "Invalid value for '{}'".format(key))

        return key, value

    def get_sequence_term_parameter(self, param_node):
        """Validate that sequence parameters are working for items in sequence."""
        key, value = self.walk([param_node.k, param_node.ast.get('v', Boolean(True))])
        if value is None:
            value = Boolean(True)

        if key != 'fork':
            raise self._error(param_node, "Unknown parameter '{}'".format(key))

        elif not isinstance(value, (Boolean, Number)) or value.value not in (True, False, 0, 1):
            raise self._error(param_node, "Invalid value for '{}'".format(key))

        return key, Boolean(bool(value.value))

    def walk__sequence(self, node):
        """Callback function to walk the AST."""
        if not self._subqueries_enabled:
            raise self._error(node, "Subqueries not supported")

        shared = []
        close = None
        params = None

        if node.ast.get('shared_by'):
            shared = self.walk(node.shared_by)

        if node.ast.get('params'):
            params = self.walk(node.params, get_param=self.get_sequence_parameter)

        # Figure out how many fields are joined by in the first query, and match across all
        first = self.walk(node.queries[0])
        num_values = len(first.join_values)
        queries = [first]  # type: list[SubqueryBy]
        queries.extend(self.walk(node.queries[1:], num_values=num_values, get_param=self.get_sequence_term_parameter))

        for query in queries:
            query.join_values = shared + query.join_values

        if node.ast.get('until'):
            close = self.walk(node.until, num_values=num_values)
            close.join_values = shared + close.join_values

        return Sequence(queries, params, close)

    # definitions
    def walk__macro(self, node):
        """Callback function to walk the AST."""
        return Macro(node.name, node.params, self.walk(node.body))

    def walk__constant(self, node):
        """Callback function to walk the AST."""
        return Constant(node.name, self.walk(node.value))


def _parse(text, start=None, preprocessor=None, implied_any=False, implied_base=False, pipes=True, subqueries=True):
    """Function for parsing EQL with arbitrary entry points.

    :param str text: EQL source text to parse
    :param str start: Entry point for the EQL grammar
    :param bool implied_any: Allow for event queries to match on any event type when a type is not specified.
         If enabled, the query ``process_name == "cmd.exe"`` becomes ``any where process_name == "cmd.exe"``
    :param bool implied_base: Allow for queries to be built with only pipes. Base query becomes 'any where true'
    :param bool pipes: Toggle support for pipes
    :param bool subqueries: Toggle support for subqueries, which are required by
        ``sequence``, ``join``, ``descendant of``, ``child of`` and ``event of``
    :param PreProcessor preprocessor: Optional preprocessor to expand definitions and constants
    :rtype: EqlNode
    """
    global GRAMMAR, tatsu_parser

    if tatsu_parser is None:
        GRAMMAR = get_etc_file('eql.ebnf')
        tatsu_parser = tatsu.compile(GRAMMAR, parseinfo=True, semantics=tatsu.semantics.ModelBuilderSemantics())

    if not text.strip():
        raise ParseError("No text specified", 0, 0, text)

    # Convert everything to unicode
    text = to_unicode(text)
    walker = EqlWalker(implied_any=implied_any, implied_base=implied_base,
                       preprocessor=preprocessor, pipes=pipes, subqueries=subqueries)

    try:
        model = tatsu_parser.parse(text, rule_name=start, start=start, parseinfo=True)
        eql_node = walker.walk(model)
        return eql_node
    except tatsu.exceptions.FailedParse as e:
        info = e.buf.line_info(e.pos)
        message = e.message
        line = info.line
        col = info.col
        source = info.text.rstrip()
        if not source:
            source = text.strip().splitlines()[-1].strip()
            col = max(len(source) - 1, 0)
        raise ParseError(message, line, col, source)


def parse_base_query(text, implied_any=False, implied_base=False, preprocessor=None, subqueries=True):
    """Parse an EQL event query without pipes.

    :param str text: EQL source text to parse
    :param bool implied_any: Allow for event queries to match on any event type when a type is not specified.
         If enabled, the query ``process_name == "cmd.exe"`` becomes ``any where process_name == "cmd.exe"``
    :param bool implied_base: Allow for queries to be built with only pipes. Base query becomes 'any where true'
    :param bool subqueries: Toggle support for subqueries, which are required by
        ``sequence``, ``join``, ``descendant of``, ``child of`` and ``event of``
    :param PreProcessor preprocessor: Optional preprocessor to expand definitions and constants
    :rtype: PipedQuery
    """
    return _parse(text, 'base_query',
                  implied_any=implied_any, implied_base=implied_base, preprocessor=preprocessor, subqueries=subqueries)


def parse_event_query(text, implied_any=False, implied_base=False, preprocessor=None, subqueries=True):
    """Parse an EQL event query in the form ``<event-type> where <condition>``.

    :param str text: EQL source text to parse
    :param bool implied_any: Allow for event queries to match on any event type when a type is not specified.
         If enabled, the query ``process_name == "cmd.exe"`` becomes ``any where process_name == "cmd.exe"``
    :param bool implied_base: Allow for queries to be built with only pipes. Base query becomes 'any where true'
    :param bool subqueries: Toggle support for subqueries, which are required by
        ``sequence``, ``join``, ``descendant of``, ``child of`` and ``event of``
    :param PreProcessor preprocessor: Optional preprocessor to expand definitions and constants
    :rtype: EventQuery
    """
    return _parse(text, 'event_query',
                  implied_any=implied_any, implied_base=implied_base, preprocessor=preprocessor, subqueries=subqueries)


def parse_query(text, implied_any=False, implied_base=False, preprocessor=None, subqueries=True, pipes=True):
    """Parse a full EQL query with pipes.

    :param str text: EQL source text to parse
    :param bool implied_any: Allow for event queries to match on any event type when a type is not specified.
         If enabled, the query ``process_name == "cmd.exe"`` becomes ``any where process_name == "cmd.exe"``
    :param bool implied_base: Allow for queries to be built with only pipes. Base query becomes 'any where true'
    :param bool subqueries: Toggle support for subqueries, which are required by
        ``sequence``, ``join``, ``descendant of``, ``child of`` and ``event of``
    :param bool pipes: Toggle support for pipes
    :param PreProcessor preprocessor: Optional preprocessor to expand definitions and constants
    :rtype: PipedQuery
    """
    return _parse(text,  'single_query', implied_any=implied_any, implied_base=implied_base, preprocessor=preprocessor,
                  subqueries=subqueries, pipes=pipes)


def parse_expression(text, implied_any=False, preprocessor=None, subqueries=True):
    """Parse an EQL expression and return the AST.

    :param str text: EQL source text to parse
    :param bool implied_any: Allow for event queries to match on any event type when a type is not specified.
         If enabled, the query ``process_name == "cmd.exe"`` becomes ``any where process_name == "cmd.exe"``
    :param bool subqueries: Toggle support for subqueries, which are required by
        ``sequence``, ``join``, ``descendant of``, ``child of`` and ``event of``
    :param PreProcessor preprocessor: Optional preprocessor to expand definitions and constants
    :rtype: Expression
    """
    return _parse(text, start='single_expression',
                  implied_any=implied_any, preprocessor=preprocessor, subqueries=subqueries)


def parse_analytic(analytic_info, preprocessor=None, **kwargs):
    """Parse an EQL analytic from a dictionary with metadata.

    :param dict analytic_info: EQL dictionary with metadata and a query to convert to an analytic.
    :param PreProcessor preprocessor: Optional preprocessor to expand definitions and constants
    :param kwargs: Additional arguments to pass to :func:`~parse_query`
    :rtype: EqlAnalytic
    """
    dct = analytic_info.copy()
    text = dct['query']
    query = parse_query(text, preprocessor=preprocessor, **kwargs)
    dct['query'] = query
    return EqlAnalytic(**dct)


def parse_analytics(analytics, preprocessor=None, **kwargs):
    """Parse EQL analytics from a list of dictionaries.

    :param list[dict] analytics: EQL dictionary with metadata to convert to an analytic.
    :param PreProcessor preprocessor: Optional preprocessor to expand definitions and constants
    :param kwargs: Additional arguments to pass to :func:`~parse_query`
    :rtype: list[EqlAnalytic]
    """
    if preprocessor is None:
        preprocessor = PreProcessor()
    return [parse_analytic(r, preprocessor=preprocessor, **kwargs) for r in analytics]


def parse_definition(text, preprocessor=None, implied_any=False, subqueries=True):
    """Parse a single EQL definition.

    :param str text: EQL source to parse
    :param PreProcessor preprocessor: Optional preprocessor to expand definitions and constants
    :param bool implied_any: Allow for event queries to skip event type and WHERE, replace with 'any where ...'
    :param bool subqueries: Toggle support for subqueries (sequence, join, named of, etc.)
    :rtype: Definition
    """
    return _parse(text, start='single_definition', preprocessor=preprocessor,
                  implied_any=implied_any, subqueries=subqueries)


def parse_definitions(text, preprocessor=None, implied_any=False, subqueries=True):
    """Parse EQL preprocessor definitions from source.

    :param str text: EQL source to parse
    :param PreProcessor preprocessor: Use an EQL preprocessor to expand definitions and constants while parsing
    :param bool implied_any: Allow for event queries to match on any event type when a type is not specified.
         If enabled, the query ``process_name == "cmd.exe"`` becomes ``any where process_name == "cmd.exe"``
    :param bool subqueries: Toggle support for subqueries, which are required by
        ``sequence``, ``join``, ``descendant of``, ``child of`` and ``event of``
    :rtype: list[Definition]
    """
    return _parse(text, start='definitions', preprocessor=preprocessor, implied_any=implied_any, subqueries=subqueries)


def get_preprocessor(text, implied_any=False, subqueries=None, preprocessor=None):
    """Parse EQL definitions and get a :class:`~eql.ast.PreProcessor`.

    :param str text: EQL source to parse
    :param PreProcessor preprocessor: Use an existing EQL preprocessor while parsing definitions
    :param bool implied_any: Allow for event queries to match on any event type when a type is not specified.
         If enabled, the query ``process_name == "cmd.exe"`` becomes ``any where process_name == "cmd.exe"``
    :param bool subqueries: Toggle support for subqueries, which are required by
        ``descendant of``, ``child of`` and ``event of``
    :rtype: PreProcessor
    """
    definitions = parse_definitions(text, implied_any=implied_any, subqueries=subqueries, preprocessor=preprocessor)

    # inherit all the definitions from the old one, and add to them
    if preprocessor is None:
        new_preprocessor = PreProcessor()
    else:
        new_preprocessor = preprocessor.copy()

    new_preprocessor.add_definitions(definitions)
    return new_preprocessor
