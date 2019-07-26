"""Python parser functions for EQL syntax."""
from __future__ import unicode_literals

import datetime
import re
import sys
from collections import OrderedDict
import threading

import tatsu
import tatsu.exceptions
import tatsu.objectmodel
import tatsu.semantics
import tatsu.walkers

from . import ast
from . import pipes
from . import types
from .errors import EqlParseError, EqlSyntaxError, EqlSemanticError, EqlSchemaError, EqlTypeMismatchError, EqlError
from .etc import get_etc_file
from .functions import get_function, list_functions
from .schema import EVENT_TYPE_ANY, EVENT_TYPE_GENERIC, Schema
from .utils import to_unicode, load_extensions, ParserConfig, is_string

__all__ = (
    "get_preprocessor",
    "parse_definition",
    "parse_definitions",
    "parse_expression",
    "parse_field",
    "parse_literal",
    "parse_query",
    "parse_analytic",
    "parse_analytics",
    "ignore_missing_fields",
    "ignore_missing_functions",
    "strict_field_schema",
    "allow_enum_fields",
)


debugger_attached = 'pydevd' in sys.modules

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

RESERVED = {n.render(): n for n in [ast.Boolean(True), ast.Boolean(False), ast.Null()]}

GRAMMAR = None
compiled_parser = None
compiler_lock = threading.Lock()

NON_SPACE_WS = re.compile(r"[^\S ]+")


ignore_missing_functions = ParserConfig(check_functions=False)
ignore_missing_fields = ParserConfig(ignore_missing_fields=False)
strict_field_schema = ParserConfig(strict_fields=True, implied_booleans=False)
allow_enum_fields = ParserConfig(enable_enum=True)


local = threading.local()

try:
    from ._parsergen import EQLParser  # noqa: E402
    local.parser = EQLParser(parseinfo=True, semantics=tatsu.semantics.ModelBuilderSemantics())
except ImportError:
    pass


def transpose(iter):
    """Transpose iterables."""
    if not iter:
        return [], []
    return [list(t) for t in zip(*iter)]


class EqlWalker(tatsu.walkers.NodeWalker):
    """Walker of Tatsu semantic model to convert it into a EQL AST."""

    def __init__(self):
        """Walker for building an EQL syntax tree from a Tatsu syntax tree.

        :param bool implied_any: Allow for event queries to skip event type and WHERE, replace with 'any where ...'
        :param bool implied_base: Allow for queries to be built with only pipes. Base query becomes 'any where true'
        :param bool subqueries: Toggle support for subqueries (sequence, join, named of, etc.)
        :param bool pipes: Toggle support for pipes
        :param PreProcessor preprocessor: Use an EQL preprocessor to expand definitions and constants while parsing
        """
        super(EqlWalker, self).__init__()
        self.implied_base = ParserConfig.read_stack("implied_base", False)
        self.implied_any = ParserConfig.read_stack("implied_any", False)

        # Create our own thread-safe copy of a preprocessor that we can use
        self.preprocessor = ParserConfig.read_stack("preprocessor", ast.PreProcessor()).copy()

        # Keep track of newly created definitions
        self.new_preprocessor = self.preprocessor.copy()

        self._subqueries_enabled = ParserConfig.read_stack("allow_subqueries", True)
        self._pipes_enabled = ParserConfig.read_stack("allow_pipes", True)
        self._function_lookup = {}

        # Allow for functions to be turned on/off and overridden
        for name in ParserConfig.read_stack("allowed_functions", list_functions()):
            self._function_lookup[name] = get_function(name)

        for signature in ParserConfig.read_stack("custom_functions", []):
            self._function_lookup[signature.name] = signature

        self._allowed_pipes = ParserConfig.read_stack("allowed_pipes", set(pipes.list_pipes()))
        self._implied_booleans = ParserConfig.read_stack("implied_booleans", True)
        self._in_pipes = False
        self._event_types = []
        self._schema = Schema.current()
        self._ignore_missing = ParserConfig.read_stack("ignore_missing_fields", False)
        self._strict_fields = ParserConfig.read_stack("strict_fields", False)
        self._allow_enum = ParserConfig.read_stack("enable_enum", False)
        self._count_keys = []
        self._pipe_schemas = []
        self._var_types = dict()
        self._check_functions = ParserConfig.read_stack("check_functions", True)

    @property
    def multiple_events(self):
        """Check if multiple events can be queried."""
        return len(self._pipe_schemas) > 1

    @property
    def event_type(self):
        """Get the active event type."""
        if not self._event_types:
            return EVENT_TYPE_ANY
        return self._event_types[-1]

    @staticmethod
    def _error(node, message, end=False, cls=EqlSemanticError, width=None, **kwargs):
        """Generate."""
        params = dict(node.ast)
        for k, value in params.items():
            if isinstance(value, list):
                params[k] = ', '.join([v.render() if isinstance(v, ast.EqlNode) else to_unicode(v) for v in value])
        params.update(kwargs)
        message = message.format(**params)
        line_number = node.parseinfo.line

        # get more lines for more informative error messages. three before + two after
        before = node.parseinfo.buffer.get_lines(0, line_number)[-3:]
        after = node.parseinfo.buffer.get_lines(line_number+1)[:2]

        source = '\n'.join(b.rstrip('\r\n') for b in before)
        trailer = '\n'.join(a.rstrip('\r\n') for a in after)

        # lines = node.parseinfo.text_lines()
        # source = '\n'.join(l.rstrip() for l in lines)
        col = node.line_info.col

        # Determine if the error message can easily look like this
        #                                                     ^^^^
        if width is None and not end:
            if not NON_SPACE_WS.search(node.text):
                width = len(node.text)

        if width is None:
            width = 1

        return cls(message, line_number, col, source, width=width, trailer=trailer)

    @classmethod
    def _type_error(cls, node, message, expected_type, actual_type=None, **kwargs):
        """Return an exception for type mismatches."""
        kwargs.setdefault('cls', EqlTypeMismatchError)
        expected_spec = types.get_specifier(expected_type)

        def get_friendly_name(t, show_spec=False):
            type_str = ""
            spec = types.get_specifier(t)

            if show_spec and spec != types.NO_SPECIFIER:
                type_str += spec + " "

            t = types.union_types(types.get_type(t))
            if not types.is_union(t):
                t = (t, )

            # now get a friendly name for all of the types
            type_strings = []
            for union_type in t:
                if isinstance(union_type, types.Nested):
                    type_strings.append("object")
                elif isinstance(union_type, types.Array):
                    if len(union_type) != 1:
                        type_strings.append("array")
                    else:
                        type_strings.append("array[{}]".format(get_friendly_name(union_type, show_spec=False)))
                elif len(t) == 1 or union_type != "null":
                    type_strings.append(to_unicode(union_type))

            return (type_str + "/".join(sorted(set(type_strings)))).strip()

        expected_type = get_friendly_name(expected_type, show_spec=True)

        if actual_type is not None:
            actual_spec = types.get_specifier(actual_type)
            spec_match = types.check_specifiers(expected_spec, actual_spec)
            expected_type = get_friendly_name(expected_type, show_spec=not spec_match)
            actual_type = get_friendly_name(actual_type, show_spec=not spec_match)

        return cls._error(node, message, actual_type=actual_type, expected_type=expected_type, **kwargs)

    def _walk_default(self, node, *args, **kwargs):
        """Callback function to walk the AST."""
        if isinstance(node, list):
            return [self.walk(n, *args, **kwargs) for n in node]
        elif isinstance(node, tuple):
            return tuple(self.walk(n, *args, **kwargs) for n in node)
        return node

    def walk(self, node, *args, **kwargs):
        """Optimize the AST while walking it."""
        event_type = kwargs.pop("event_type", None)
        split = kwargs.pop("split", False)

        if event_type is not None:
            self._event_types.append(event_type)

        output = super(EqlWalker, self).walk(node, *args, **kwargs)

        if event_type is not None:
            self._event_types.pop()

        if isinstance(output, tuple) and isinstance(output[0], ast.EqlNode) and isinstance(output[1], tuple):
            output_node, output_hint = output
            output_node = output_node.optimize()

            # If it was optimized to a literal, the type may be constrained
            if isinstance(output_node, ast.Literal):
                output_hint = types.get_specifier(output_hint), types.get_type(output_node.type_hint)

            output = output_node, output_hint
        elif isinstance(output, ast.EqlNode):
            return output.optimize()

        if split:
            if isinstance(output, list):
                return [list(o) for o in transpose(output)]
            return zip(*output)

        return output

    def validate_signature(self, node, signature, arguments, hints):
        """Validate a signature against input arguments and type hints."""
        error_node = node
        node_type = 'pipe' if issubclass(signature, ast.PipeCommand) else 'function'
        name = signature.name
        bad_index, new_arguments, new_hints = signature.validate(arguments, hints)

        if bad_index is None:
            # no error exists, so no need to build a message
            return new_arguments, new_hints

        min_args = signature.minimum_args if signature.minimum_args is not None else len(signature.argument_types)
        max_args = None

        if signature.additional_types is None:
            max_args = len(signature.argument_types)

        # Try to line up the error message with the argument that went wrong
        # Strings and numbers don't generate tatsu nodes, so its difficult to recover parseinfo
        if min_args is not None and len(arguments) < min_args:
            message = "Expected at least {} argument{} to pipe {}".format(
                min_args, 's' if min_args != 1 else '', node.name)
            raise self._error(error_node, message, end=len(arguments) != 0)

        elif max_args is not None and max_args < len(arguments):
            if max_args == 0:
                argument_desc = 'no arguments'
            elif max_args == 1:
                argument_desc = 'only 1 argument'
            else:
                argument_desc = 'up to {} arguments'.format(max_args)
            message = "Expected {} to {} {}".format(argument_desc, node_type, name)
            error_node = node.args[max_args]
            raise self._error(error_node, message)

        elif bad_index is not None:
            if isinstance(node.args[bad_index], tatsu.semantics.Node):
                error_node = node.args[bad_index]

            actual_type = hints[bad_index]
            expected_type = signature.additional_types

            if bad_index < len(signature.argument_types):
                expected_type = signature.argument_types[bad_index]

            if expected_type is not None and not types.check_full_hint(expected_type, actual_type):
                raise self._type_error(error_node, "Expected {expected_type} not {actual_type} to {name}",
                                       expected_type, actual_type, name=name)
            raise self._error(error_node, "Invalid argument to {name}", name=name)

        return new_arguments, new_hints

    def walk__root_expression(self, node, keep_hint=False, query_condition=False):
        """Get the root expression, and rip out the type hint."""
        expr, hint = self.walk(node.expr)
        if query_condition and not self._implied_booleans and not types.check_types(types.BOOLEAN, hint):
            raise self._type_error(node.expr, "Expected {expected_type} not {actual_type}", types.BOOLEAN, hint)
        if keep_hint:
            return expr, hint
        return expr

    # literals
    def walk__literal(self, node, **kwargs):
        """Callback function to walk the AST."""
        value = self.walk(node.value)
        cls = ast.Literal.find_type(value)

        if cls is ast.String:
            value = to_unicode(value)

            # If a 'raw' string is detected, then only unescape the quote character
            if node.text.startswith('?'):
                quote_char = node.text[-1]
                value = value.replace("\\" + quote_char, quote_char)
            else:
                value = ast.String.unescape(value)

        return cls(value), types.literal(cls.type_hint)

    def walk__time_range(self, node):
        """Callback function to walk the AST."""
        val = self.walk(node.val)
        unit = self.walk(node.unit)
        for name, interval in units.items():
            if name.startswith(unit.rstrip('s') or 's'):
                return ast.TimeRange(datetime.timedelta(seconds=val * interval)), types.literal(types.NUMBER)

        raise self._error(node.unit, "Unknown time unit")

    def walk__check_parentheses(self, node):
        """Check that parentheses are matching."""
        # check for the deepest one first, so it can raise an exception
        expr = self.walk(node.expr)

        if node.ast.get('closing', ')') is None:
            raise self._error(node, "Mismatched parentheses ()")
        return expr

    # fields
    def walk__attribute(self, node):
        """Validate attributes."""
        if node.attr in RESERVED:
            raise self._error(node, "Illegal use of reserved value")
        return node.attr

    def walk__array_index(self, node):
        """Get the index for the field in the array."""
        if node.ast.get('value', None) is not None:
            return node.value

        if node.ast.get('closing', ']') is None:
            raise self._error(node, "Mismatched brackets []")

        if 'missing' in node.ast:
            raise self._error(node, "Required index to array.")
        raise self._error(node, "Invalid index to array.")

    def _get_field_hint(self, node, field, allow_enum=False):
        type_hint = types.BASE_ALL
        allow_missing = self._schema.allow_missing

        if self._in_pipes:
            event_schema = self._pipe_schemas[0]
            event_field = field
            if self.multiple_events:
                event_index, event_field = field.query_multiple_events()
                num_events = len(self._pipe_schemas)
                if event_index >= num_events:
                    raise self._error(node.sub_fields[0], "Invalid index. Event array is size {num}", num=num_events)
                event_schema = self._pipe_schemas[event_index]

            # Now that we have the schema
            event_type, = event_schema.schema.keys()
            type_hint = event_schema.get_event_type_hint(event_type, event_field.full_path)
            allow_missing = self._schema.allow_missing

        elif not self._schema:
            return field, types.dynamic(type_hint)

        # check if it's a variable and
        elif node.base not in self._var_types:
            event_field = field
            event_type = self.event_type

            type_hint = self._schema.get_event_type_hint(event_type, event_field.full_path)

            # Determine if the field should be converted as an enum
            # from subtype.create -> subtype == "create"
            if type_hint is None and self._allow_enum and event_field.path and is_string(event_field.path[-1]):
                base_field = ast.Field(event_field.base, event_field.path[:-1])
                enum_value = ast.String(event_field.path[-1])
                base_hint = self._schema.get_event_type_hint(event_type, base_field.full_path)

                if types.check_types(types.STRING, base_hint):
                    return ast.Comparison(base_field, ast.Comparison.EQ, enum_value), types.dynamic(types.BOOLEAN)

        if type_hint is None and not allow_missing:
            message = "Field not recognized"
            if event_type not in (EVENT_TYPE_ANY, EVENT_TYPE_GENERIC):
                message += " for {event_type} event"
            raise self._error(node, message, cls=EqlSchemaError, event_type=event_type)

        # the field could be missing, so allow for null checks unless it's explicitly disabled
        if not self._strict_fields:
            type_hint = types.union(type_hint, types.NULL)

        return field, types.dynamic(type_hint)

    def walk__field(self, node, get_variable=False, **kwargs):
        """Callback function to walk the AST."""
        if get_variable:
            if node.base in RESERVED or node.sub_fields:
                raise self._type_error(node, "Expected {expected_type} not {field} to function", types.VARIABLE)
            elif node.base in self._var_types:
                raise self._error(node, "Reuse of variable {base}")

            # This can be overridden by the parent function that is parsing it
            self._var_types[node.base] = types.BASE_ALL
            return ast.Field(node.base), types.VARIABLE

        if node.base in RESERVED:
            if len(node.sub_fields) != 0:
                raise self._error(node, "Illegal use of reserved value")

            value = RESERVED[node.base]
            return value, types.literal(value.type_hint)

        path = self.walk(node.sub_fields)

        if not path and node.base in self.preprocessor.constants:
            constant = self.preprocessor.constants[node.base]
            return constant.value, types.literal(constant.value.type_hint)

        # Check if it's part of the current preprocessor that we are building
        # and if it is, then return it unexpanded but with a type hint
        if not path and node.base in self.new_preprocessor.constants:
            constant = self.new_preprocessor.constants[node.base]
            return ast.Field(node.base), types.literal(constant.value.type_hint)

        field = ast.Field(node.base, path)
        return self._get_field_hint(node, field, allow_enum=self._allow_enum)

    # comparisons
    def walk__equals(self, node):
        """Callback function to walk the AST."""
        # May be double or single equals
        return '=='

    def walk__comparator(self, node):
        """Walk comparators like <= < != == > >=."""
        return self.walk(node.comp)

    def walk__comparison(self, node):
        """Callback function to walk the AST."""
        left, left_type = self.walk(node.left)
        right, right_type = self.walk(node.right)
        op = self.walk(node.op)

        accepted_types = types.union(types.PRIMITIVES, types.NULL)
        error_message = "Unable to compare {expected_type} to {actual_type}"

        if not types.check_types(left_type, right_type) or \
                not types.check_types(accepted_types, left_type) or \
                not types.check_types(accepted_types, right_type):
            # check if the types can actually be compared, and don't allow comparison of nested types
            raise self._type_error(node.op, error_message, types.clear(left_type), types.clear(right_type))

        if op in (ast.Comparison.LT, ast.Comparison.LE, ast.Comparison.GE, ast.Comparison.GE):
            # check that <, <=, >, >= are only supported for strings or integers
            lt = types.get_type(left_type)
            rt = types.get_type(right_type)

            # string to string or number to number
            if not ((types.check_full_hint(types.STRING, lt) and types.check_full_hint(types.STRING, rt)) or
                    (types.check_full_hint(types.NUMBER, lt) and types.check_full_hint(types.NUMBER, rt))):
                raise self._type_error(node.op, error_message, types.clear(left_type), types.clear(right_type))

        comp_node = ast.Comparison(left, op, right)
        hint = types.get_specifier(types.union(left_type, right_type)), types.get_type(types.BOOLEAN)

        # there is no special comparator for wildcards, just look for * in the string
        if isinstance(right, ast.String) and '*' in right.value:
            func_call = ast.FunctionCall('wildcard', [left, right])

            if op == ast.Comparison.EQ:
                return func_call, hint
            elif op == ast.Comparison.NE:
                return ~ func_call, hint

        return comp_node, hint

    def walk__and_terms(self, node):
        """Callback function to walk the AST."""
        terms, hints = self.walk(node.terms, split=True)
        if not self._implied_booleans:
            for tatsu_node, hint in zip(node.terms, hints):
                if not types.check_types(types.BOOLEAN, hint):
                    raise self._type_error(tatsu_node, "Expected {expected_type}, not {actual_type}",
                                           types.BOOLEAN, hint)

        term = ast.And(terms)
        return term, types.union(*hints)

    def walk__or_terms(self, node):
        """Callback function to walk the AST."""
        terms, hints = self.walk(node.terms, split=True)
        if not self._implied_booleans:
            for tatsu_node, hint in zip(node.terms, hints):
                if not types.check_types(types.BOOLEAN, hint):
                    raise self._type_error(tatsu_node, "Expected {expected_type}, not {actual_type}",
                                           types.BOOLEAN, hint)
        term = ast.Or(terms)
        return term, types.union(*hints)

    def walk__not_term(self, node):
        """Callback function to walk the AST."""
        term, hint = self.walk(node.t)
        return ~ term, types.union(hint)

    def walk__in_set(self, node):
        """Callback function to walk the AST."""
        expr, outer_hint = self.walk(node.expr)
        container, sub_hints = self.walk(node.container, keep_hint=True, split=True)
        outer_spec = types.get_specifier(outer_hint)
        outer_type = types.get_type(outer_hint)
        container_specifiers = [types.get_specifier(h) for h in sub_hints]
        container_types = [types.get_type(h) for h in sub_hints]

        # Check that everything inside the container has the same type as outside
        error_message = "Unable to compare {expected_type} to {actual_type}"
        for container_node, node_type in zip(node.container, container_types):
            if not types.check_types(outer_type, node_type):
                raise self._type_error(container_node, error_message, outer_type, node_type)

        # This will always evaluate to true/false, so it should be a boolean
        term = ast.InSet(expr, container)
        return term, (types.union_specifiers(outer_spec, *container_specifiers), types.BASE_BOOLEAN)

    def _get_type_hint(self, node, ast_node):
        """Get the recommended type hint for a node when it isn't already known.

        This will likely only get called when expanding macros, until type hints are attached to AST nodes.
        """
        type_hint = types.EXPRESSION

        if isinstance(ast_node, ast.Literal):
            type_hint = ast_node.type_hint
        elif isinstance(ast_node, (ast.Comparison, ast.InSet)):
            type_hint = types.BOOLEAN
        elif isinstance(ast_node, ast.Field):
            type_hint = types.EXPRESSION

            if ast_node.base not in self._var_types:
                ast_node, type_hint = self._get_field_hint(node, ast_node)

            # Make it dynamic because it's a field
            type_hint = types.dynamic(type_hint)

            if not self._strict_fields:
                type_hint = types.union(type_hint, types.NULL)

        elif isinstance(ast_node, ast.FunctionCall):
            signature = self._function_lookup.get(node.name)
            if signature:
                type_hint = signature.return_value

        if any(isinstance(n, ast.Field) for n in ast_node):
            type_hint = types.dynamic(type_hint)

        return type_hint

    def walk__function_call(self, node):
        """Callback function to walk the AST."""
        if node.name in self.preprocessor.macros:
            args = []

            if node.args:
                args, hints = self.walk(node.args, split=True)

            macro = self.preprocessor.macros[node.name]
            expanded = macro.expand(args)
            type_hint = self._get_type_hint(node, expanded)
            return expanded, type_hint

        elif node.name in self.new_preprocessor.macros:
            args = []

            if node.args:
                args, hints = self.walk(node.args, split=True)
            macro = self.new_preprocessor.macros[node.name]
            expanded = macro.expand(args)
            type_hint = self._get_type_hint(node, expanded)
            return expanded, type_hint

        signature = self._function_lookup.get(node.name)

        if signature:
            # Check for any variables in the signature, and handle their type hints differently
            variables = set(idx for idx, hint in enumerate(signature.argument_types) if hint == types.VARIABLE)

            arguments = []

            # Back up the current variable type hints for when this function goes out of scope
            old_variables = self._var_types.copy()

            # Get all of the arguments first, because they may depend on others
            # and we need to pull out all of the variables
            for idx, arg_node in enumerate(node.args or []):
                if idx in variables:
                    exc = self._type_error(arg_node, "Invalid argument to {name}. Expected {expected_type}",
                                           types.VARIABLE, name=node.name)

                    if arg_node.parseinfo.rule == 'field':
                        try:
                            arguments.append(self.walk(arg_node, get_variable=True))
                        except EqlTypeMismatchError:
                            pass
                        else:
                            continue

                    # Ignore the original exception and raise our own, which has the function name in it
                    raise exc

                else:
                    arguments.append(self.walk(arg_node))

            # Then validate this against the signature
            args, hints = transpose(arguments)

            # In theory, we could do another round of validation for generics, but we'll just assume
            # that loop variables can take any shape they need to, as long as the other arguments match

            # Validate that the arguments match the function signature by type and length
            args, hints = self.validate_signature(node, signature, args, hints)

            # Restore old variables, since ours are out of scope now
            self._var_types = old_variables

            # Get return value and specifier, and mark as dynamic if any of the inputs are
            output_hint = signature.return_value

            if hints and types.is_dynamic(types.union(*hints)):
                output_hint = types.dynamic(output_hint)

            return ast.FunctionCall(node.name, args), output_hint

        elif self._check_functions:
            raise self._error(node, "Unknown function {name}", width=len(node.name))
        else:
            args = []

            if node.args:
                args, _ = self.walk(node.args, split=True)

            return ast.FunctionCall(node.name, args), types.dynamic(types.EXPRESSION)

    # queries
    def walk__event_query(self, node):
        """Callback function to walk the AST."""
        if node.ast.get('event_type') is None:
            event_type = EVENT_TYPE_ANY
            if not self.implied_any:
                raise self._error(node, "Missing event type and 'where' condition")
        else:
            event_type = node.event_type

            if self._schema and not self._schema.validate_event_type(event_type):
                raise self._error(node, "Invalid event type: {event_type}", cls=EqlSchemaError, width=len(event_type))

        condition = self.walk(node.cond, event_type=event_type, query_condition=True)
        return ast.EventQuery(event_type, condition)

    def walk__pipe(self, node):
        """Callback function to walk the AST."""
        if not self._pipes_enabled:
            raise self._error(node, "Pipes not supported")

        pipe_cls = ast.PipeCommand.lookup.get(node.name)
        if pipe_cls is None or node.name not in self._allowed_pipes:
            raise self._error(node, "Unknown pipe {name}", width=len(node.name))

        args = []
        hints = []

        if node.args:
            args, hints = self.walk(node.args, split=True)

        args, hints = self.validate_signature(node, pipe_cls, args, hints)
        self._pipe_schemas = pipe_cls.output_schemas(args, hints, self._pipe_schemas)
        return pipe_cls(args)

    def walk__piped_query(self, node):
        """Callback function to walk the AST."""
        if node.query is None:
            first = ast.EventQuery(EVENT_TYPE_ANY, ast.Boolean(True))
            if not self.implied_base:
                raise self._error(node, "Missing base query")
        else:
            first = self.walk(node.query)

        self._in_pipes = True
        if isinstance(first, ast.EventQuery):
            base_event_types = [first.event_type]
        else:
            base_event_types = [q.query.event_type for q in first.queries]

        # Now, create the schema for each event in the array
        flattened_schema = self._schema.flatten()
        for event_type in base_event_types:
            if event_type == EVENT_TYPE_ANY:
                self._pipe_schemas.append(flattened_schema)
            elif event_type in self._schema.schema:
                self._pipe_schemas.append(Schema({EVENT_TYPE_GENERIC: self._schema.schema[event_type]}))
            else:
                self._pipe_schemas.append(Schema({EVENT_TYPE_GENERIC: {}}))

        return ast.PipedQuery(first, self.walk(node.pipes))

    def walk__subquery_type(self, node):
        """Get the subquery type."""
        if not self._subqueries_enabled:
            raise self._error(node, "Subqueries not supported")
        elif self._in_pipes:
            raise self._error(node, "Not supported within pipe")

        if node.name not in ast.NamedSubquery.supported_types:
            raise self._error(node, "Unknown subquery type '{name} of'")

        return node.name

    def walk__named_query(self, node):
        """Callback function to walk the AST."""
        return ast.NamedSubquery(self.walk(node.stype), self.walk(node.query)), types.dynamic(types.BOOLEAN)

    def walk__named_params(self, node, get_param=None, position=None, close=None):
        """Callback function to walk the AST."""
        params = OrderedDict()
        if get_param is None and len(node.params) > 0:
            raise self._error(node, "Unexpected parameters")

        for param in node.params:
            key, value = get_param(param, position=position, close=close)
            if key in params:
                raise self._error(param, "Repeated parameter {k}")
            params[key] = value
        return ast.NamedParams(params)

    def walk__subquery_by(self, node, num_values=None, position=None, close=None, get_param=None):
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
            message = "Expected {num} value"
            if num_values != 1:
                message += "s"
            raise self._error(error_node, message, num=num_values, end=end)

        params = self.walk(node.params, get_param=get_param, position=position, close=close)
        query = self.walk(node.query)
        if node.join_values:
            join_values, join_hints = self.walk(node.join_values, event_type=query.event_type, split=True)
        else:
            join_values, join_hints = [], []
        return ast.SubqueryBy(query, params, join_values), join_hints

    def walk__join(self, node):
        """Callback function to walk the AST."""
        queries, close = self._get_subqueries_and_close(node)
        return ast.Join(queries, close)

    def _get_subqueries_and_close(self, node, get_param=None):
        """Helper function used by join and sequence to avoid duplicate code."""
        if not self._subqueries_enabled:
            # Raise the error earlier (instead of waiting until subquery_by) so that it's more meaningful
            raise self._error(node, "Subqueries not supported")

        # Figure out how many fields are joined by in the first query, and match across all
        first, first_hints = self.walk(node.queries[0], get_param=get_param, position=0)
        num_values = len(first.join_values)
        queries = [(first, first_hints)]

        for pos, query in enumerate(node.queries[1:], 1):
            queries.append(self.walk(query, num_values=num_values, get_param=get_param, position=pos))

        shared = node.ast.get('shared_by')
        close = None

        # Validate that each field has matching types
        default_hint = types.get_type(types.union(types.PRIMITIVES, types.NULL))
        strict_hints = [default_hint] * num_values

        if shared:
            strict_hints += [default_hint] * len(shared)

        def check_by_field(by_pos, by_node, by_hint):
            # Check that the possible values for our field that match what we currently understand about this type
            intersected = types.intersect_types(strict_hints[by_pos], by_hint)
            if not intersected or not types.is_dynamic(by_hint):
                raise self._type_error(by_node, "Unable to join {expected_type} to {actual_type}",
                                       strict_hints[by_pos], by_hint)

            # Restrict the acceptable fields from what we've seen
            strict_hints[by_pos] = intersected

        for qpos, (query, query_by_hints) in enumerate(queries):
            unshared_fields = []
            curr_by_hints = query_by_hints
            curr_join_nodes = node.queries[qpos].join_values

            if shared:
                curr_shared_by, curr_shared_hints = self.walk(shared, event_type=query.query.event_type, split=True)
                curr_by_hints = curr_shared_hints + curr_by_hints
                query.join_values = curr_shared_by + query.join_values
                curr_join_nodes = shared + curr_join_nodes

            # Now that they've all been built out, start to intersect the types
            for fpos, (n, h) in enumerate(zip(curr_join_nodes, curr_by_hints)):
                check_by_field(fpos, n, h)

            # Add all of the fields to the beginning of this subquery's BY fields and preserve the order
            query.join_values = unshared_fields + query.join_values

        if node.ast.get("until"):
            close, close_hints = self.walk(node.until, num_values=num_values, get_param=get_param, close=True)
            close_nodes = [node.until]

            if shared:
                shared_by, shared_hints = self.walk(node.shared_by, event_type=close.query.event_type, split=True)
                close_hints = close_hints + shared_hints
                close.join_values = shared_by + close.join_values
                close_nodes = shared + close_nodes

            # Check the types of the by field
            for fpos, (n, h) in enumerate(zip(close_nodes, close_hints)):
                check_by_field(fpos, n, h)

        # Unzip the queries from the (query, hint) tuples
        queries, _ = zip(*queries)
        return list(queries), close

    def get_sequence_parameter(self, node, **kwargs):
        """Validate that sequence parameters are working."""
        key, (value, value_hint) = self.walk([node.k, node.v])
        value = ast.TimeRange.convert(value)

        if key != 'maxspan':
            raise self._error(node, "Unknown sequence parameter {}".format(key))

        if not ast.TimeRange.convert(value) or value.delta < datetime.timedelta(0):
            error_node = node.v if isinstance(node.v, tatsu.objectmodel.Node) else node
            raise self._error(error_node, "Invalid value for {}".format(key))

        return key, value

    def get_sequence_term_parameter(self, param_node, position, close):
        """Validate that sequence parameters are working for items in sequence."""
        if not position or close:
            raise self._error(param_node, "Unexpected parameters")

        # set the default type to a literal 'true'
        value, type_hint = ast.Boolean(True), types.literal(types.BOOLEAN)
        key = self.walk(param_node.k)
        if param_node.ast.get('v'):
            value, type_hint = self.walk(param_node.v)

        if key == 'fork':
            if not types.check_types(types.literal((types.NUMBER, types.BOOLEAN)), type_hint):
                raise self._type_error(param_node,
                                       "Expected type {expected_type} value for {k}",
                                       types.literal(types.BOOLEAN))

            if value.value not in (True, False, 0, 1):
                raise self._error(param_node, "Invalid value for {k}")

        else:
            raise self._error(param_node, "Unknown parameter {k}")

        return key, ast.Boolean(bool(value.value))

    def walk__sequence(self, node):
        """Callback function to walk the AST."""
        if not self._subqueries_enabled:
            raise self._error(node, "Subqueries not supported")

        params = None

        if node.ast.get('params'):
            params = self.walk(node.params, get_param=self.get_sequence_parameter)

        queries, close = self._get_subqueries_and_close(node, get_param=self.get_sequence_term_parameter)
        return ast.Sequence(queries, params, close)

    # definitions
    def walk__macro(self, node):
        """Callback function to walk the AST."""
        definition = ast.Macro(node.name, node.params, self.walk(node.body))
        self.new_preprocessor.add_definition(definition)
        return definition

    def walk__constant(self, node):
        """Callback function to walk the AST."""
        value, _ = self.walk(node.value)
        definition = ast.Constant(node.name, value)
        self.new_preprocessor.add_definition(definition)
        return definition


def _build_parser():
    """Build a parser one-time. These appear to be thread-safe so this only needs to happen once."""
    global GRAMMAR, compiled_parser

    if compiled_parser is not None:
        return compiled_parser

    with compiler_lock:
        if compiled_parser is None:
            GRAMMAR = get_etc_file('eql.ebnf')
            compiled_parser = tatsu.compile(GRAMMAR, parseinfo=True, semantics=tatsu.semantics.ModelBuilderSemantics())

    return compiled_parser


def _get_parser():
    """Try to get a thread-safe parser, and compile if necessary."""
    if not hasattr(local, "parser"):
        local.parser = _build_parser()
    return local.parser


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
    parser = _get_parser()

    if not text.strip():
        raise EqlParseError("No text specified", 0, 0, text)

    # Convert everything to unicode
    text = to_unicode(text)

    with ParserConfig(implied_any=implied_any, implied_base=implied_base, allow_subqueries=subqueries,
                      preprocessor=preprocessor, allow_pipes=pipes) as cfg:

        walker = EqlWalker()
        load_extensions(force=False)
        exc = None

        try:
            model = parser.parse(text, rule_name=start, start=start, parseinfo=True)
            eql_node = walker.walk(model)
            if not isinstance(eql_node, ast.EqlNode) and isinstance(eql_node, tuple):
                eql_node, type_hint = eql_node
            return eql_node
        except EqlError as e:
            # If full traceback mode is enabled, then re-raise the exception
            if cfg.read_stack("full_traceback", debugger_attached):
                raise
            exc = e
        except tatsu.exceptions.FailedParse as e:
            # Remove the tatsu exception from the traceback
            exc = e

    if isinstance(exc, EqlError):
        # at this point, the full traceback isn't wanted, so raise it from here
        raise exc

    if isinstance(exc, tatsu.exceptions.FailedParse):
        info = exc.buf.line_info(exc.pos)
        message = 'Invalid syntax'
        line = info.line
        col = info.col

        source = info.text.rstrip()
        if not source:
            source = text.rstrip().splitlines()[-1].rstrip()
            col = max(len(source) - 1, 0)

        # Raise an EQL error instead
        raise EqlSyntaxError(message, line, col, source)


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


def parse_query(text, implied_any=False, implied_base=False, preprocessor=None, subqueries=True, pipes=True, cli=False):
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
    rule = "cli_query" if cli else "single_query"
    return _parse(text,  rule, implied_any=implied_any, implied_base=implied_base, preprocessor=preprocessor,
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


def parse_atom(text, cls=None):  # type: (str, type) -> ast.Field|ast.Literal
    """Parse and get an atom."""
    rule = "single_atom"
    atom = _parse(text, start="single_atom")
    if cls is not None and not isinstance(atom, cls):
        walker = EqlWalker()
        tatsu_ast = _get_parser().parse(text, rule_name=rule, start=rule, parseinfo=True)
        raise walker._error(tatsu_ast, "Expected {expected} not {actual}",
                            expected=cls.__name__.lower(), actual=type(atom).__name__.lower())
    return atom


def parse_literal(text):  # type: (str) -> ast.Literal
    """Parse and get a literal."""
    return parse_atom(text, cls=ast.Literal)


def parse_field(text):  # type: (str) -> ast.Field
    """Parse and get a field."""
    return parse_atom(text, cls=ast.Field)


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
    return ast.EqlAnalytic(**dct)


def parse_analytics(analytics, preprocessor=None, **kwargs):
    """Parse EQL analytics from a list of dictionaries.

    :param list[dict] analytics: EQL dictionary with metadata to convert to an analytic.
    :param PreProcessor preprocessor: Optional preprocessor to expand definitions and constants
    :param kwargs: Additional arguments to pass to :func:`~parse_query`
    :rtype: list[EqlAnalytic]
    """
    if preprocessor is None:
        preprocessor = ast.PreProcessor()
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
        new_preprocessor = ast.PreProcessor()
    else:
        new_preprocessor = preprocessor.copy()

    new_preprocessor.add_definitions(definitions)
    return new_preprocessor
