"""Python parser functions for EQL syntax."""
from __future__ import unicode_literals

import datetime
import re
import sys
from collections import OrderedDict, defaultdict
import contextlib

from lark import Lark, Tree, Token
from lark.visitors import Interpreter
from lark.exceptions import LarkError

from . import ast
from . import pipes
from . import types
from .errors import EqlSyntaxError, EqlSemanticError, EqlSchemaError, EqlTypeMismatchError, EqlError
from .etc import get_etc_file
from .functions import get_function, list_functions
from .optimizer import Optimizer
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
    "skip_optimizations",
    "strict_field_schema",
    "allow_enum_fields",
    "extract_query_terms",
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


NON_SPACE_WS = re.compile(r"[^\S ]+")


skip_optimizations = ParserConfig(optimized=False)
ignore_missing_functions = ParserConfig(check_functions=False)
ignore_missing_fields = ParserConfig(ignore_missing_fields=False)
strict_field_schema = ParserConfig(strict_fields=True, implied_booleans=False)
allow_enum_fields = ParserConfig(enable_enum=True)

keywords = ("and", "by", "const", "false", "in", "join", "macro",
            "not", "null", "of", "or", "sequence", "true", "until", "with", "where"
            )


class KvTree(Tree):
    """Helper class with methods for looking up child nodes by name."""

    def get(self, name):
        """Get a child by the name of the data."""
        for match in self.get_list(name):
            return match

    def get_list(self, name):
        """Get a list of all children for a name."""
        return [child for child in self.children
                if isinstance(child, Token) and child.type == name or
                isinstance(child, KvTree) and child.data == name]

    def __contains__(self, item):
        return any(isinstance(child, Token) and child.type == item or
                   isinstance(child, KvTree) and child.data == item for child in self.children)

    def __getitem__(self, item):
        """Helper method for getting by index."""
        return self.get(item)

    @property
    def child_trees(self):
        return [child for child in self.children if isinstance(child, KvTree)]


class LarkToEQL(Interpreter):
    """Walker of Lark tree to convert it into a EQL AST."""

    def __init__(self, text):
        """Walker for building an EQL syntax tree from a Lark tree.

        :param bool implied_any: Allow for event queries to skip event type and WHERE, replace with 'any where ...'
        :param bool implied_base: Allow for queries to be built with only pipes. Base query becomes 'any where true'
        :param bool subqueries: Toggle support for subqueries (sequence, join, named of, etc.)
        :param bool pipes: Toggle support for pipes
        :param PreProcessor preprocessor: Use an EQL preprocessor to expand definitions and constants while parsing
        """
        self.text = text
        self._lines = None
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
        self._stacks = defaultdict(list)

    @property
    def lines(self):
        """Lazily split lines in the original text."""
        if self._lines is None:
            self._lines = [t.rstrip("\r\n") for t in self.text.splitlines(True)]

        return self._lines

    @staticmethod
    def unzip_hints(rv):
        """Separate a list of (node, hints) into separate arrays."""
        rv = list(rv)

        if not rv:
            return [], []

        nodes, hints = zip(*rv)
        return list(nodes), list(hints)

    @contextlib.contextmanager
    def scoped(self, **kv):
        """Set scoped values."""
        for k, v in kv.items():
            self._stacks[k].append(v)
        try:
            yield
        finally:
            for k in kv:
                self._stacks[k].pop()

    def scope(self, name, default=None):
        """Read something from the scope."""
        stack = self._stacks[name]
        if len(stack) == 0:
            return default
        return stack[-1]

    @property
    def multiple_events(self):
        """Check if multiple events can be queried."""
        return len(self._pipe_schemas) > 1

    def _error(self, node, message, end=False, cls=EqlSemanticError, width=None, **kwargs):
        # type: (KvTree, str, bool, type, int, object) -> Exception
        """Generate."""
        params = {}
        for child in node.children:
            if isinstance(child, Token):
                params[child.type] = child.value
            elif isinstance(child, KvTree):
                # TODO: Recover the original string slice
                params[child.data] = child

        for k, value in params.items():
            if isinstance(value, list):
                params[k] = ', '.join([v.render() if isinstance(v, ast.EqlNode) else to_unicode(v) for v in value])

        params.update(kwargs)
        message = message.format(**params)
        line_number = node.line - 1 if not end else node.end_line - 1
        column = node.column - 1 if not end else node.end_column - 1

        # get more lines for more informative error messages. three before + two after
        before = self.lines[:line_number + 1][-3:]
        after = self.lines[line_number + 1:][:3]

        source = '\n'.join(b for b in before)
        trailer = '\n'.join(a for a in after)

        # lines = node.parseinfo.text_lines()
        # source = '\n'.join(l.rstrip() for l in lines)

        # Determine if the error message can easily look like this
        #                                                     ^^^^
        if width is None and not end and node.line == node.end_line:
            if not NON_SPACE_WS.search(self.lines[line_number][column:node.end_column]):
                width = node.end_column - node.column

        if width is None:
            width = 1

        return cls(message, line_number, column, source, width=width, trailer=trailer)

    def _type_error(self, node, message, expected_type, actual_type=None, **kwargs):
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
                        type_strings.append("array[{}]".format(get_friendly_name(union_type[0], show_spec=False)))
                elif len(t) == 1 or union_type != "null":
                    type_strings.append(to_unicode(union_type))

            return (type_str + "/".join(sorted(set(type_strings)))).strip()

        expected_type = get_friendly_name(expected_type, show_spec=True)

        if actual_type is not None:
            actual_spec = types.get_specifier(actual_type)
            spec_match = types.check_specifiers(expected_spec, actual_spec)
            expected_type = get_friendly_name(expected_type, show_spec=not spec_match)
            actual_type = get_friendly_name(actual_type, show_spec=not spec_match)

        return self._error(node, message, actual_type=actual_type, expected_type=expected_type, **kwargs)

    def _walk_default(self, node, *args, **kwargs):
        """Callback function to walk the AST."""
        if isinstance(node, list):
            return [self.walk(n, *args, **kwargs) for n in node]
        elif isinstance(node, tuple):
            return tuple(self.walk(n, *args, **kwargs) for n in node)
        return node

    def visit_children(self, tree):
        """Wrap visit_children to be more flexible."""
        if tree is None:
            return None

        return Interpreter.visit_children(self, tree)

    def visit(self, tree):
        """Optimize a return value."""
        if tree is None:
            return None

        if isinstance(tree, list):
            return [self.visit(t) for t in tree]

        rv = Interpreter.visit(self, tree)

        if isinstance(rv, tuple) and rv and isinstance(rv[0], ast.EqlNode):
            output_node, output_hint = rv

            # If it was optimized to a literal, the type may be constrained
            if isinstance(output_node, ast.Literal):
                output_hint = types.get_specifier(output_hint), types.get_type(output_node.type_hint)

            return output_node, output_hint
        return rv

    def validate_signature(self, node, signature, argument_nodes, arguments, hints):
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
        if min_args is not None and len(arguments) < min_args:
            message = "Expected at least {} argument{} to {} {}".format(
                min_args, 's' if min_args != 1 else '', node_type, self.visit(node["name"]))
            raise self._error(error_node, message, end=len(arguments) != 0)

        elif max_args is not None and max_args < len(arguments):
            if max_args == 0:
                argument_desc = 'no arguments'
            elif max_args == 1:
                argument_desc = 'only 1 argument'
            else:
                argument_desc = 'up to {} arguments'.format(max_args)
            message = "Expected {} to {} {}".format(argument_desc, node_type, name)
            error_node = argument_nodes[max_args]
            raise self._error(error_node, message)

        elif bad_index is not None:
            if isinstance(argument_nodes[bad_index], (KvTree, Token)):
                error_node = argument_nodes[bad_index]

            actual_type = hints[bad_index]
            expected_type = signature.additional_types

            if bad_index < len(signature.argument_types):
                expected_type = signature.argument_types[bad_index]

            if expected_type is not None and not types.check_full_hint(expected_type, actual_type):
                raise self._type_error(error_node, "Expected {expected_type} not {actual_type} to {name}",
                                       expected_type, actual_type, name=name)
            raise self._error(error_node, "Invalid argument to {name}", name=name)

        return new_arguments, new_hints

    def start(self, node):
        """Entry point for the grammar."""
        return self.visit(node.children[0])

    # literals
    def literal(self, node):
        """Callback function to walk the AST."""
        value, = self.visit_children(node)
        if is_string(value):
            return ast.String(value), types.literal(ast.String)
        return ast.Number(value), types.literal(ast.Number.type_hint)

    def time_range(self, node):
        """Callback function to walk the AST."""
        val, unit = self.visit_children(node)

        for name, interval in units.items():
            if name.startswith(unit.rstrip('s') or 's'):
                return ast.TimeRange(datetime.timedelta(seconds=val * interval)), types.literal(types.NUMBER)

        raise self._error(node["name"], "Unknown time unit")

    # fields
    def _get_field_hint(self, node, field):
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
        elif field.base not in self._var_types:
            event_field = field
            event_type = self.scope("event_type", default=EVENT_TYPE_ANY)

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

    def _add_variable(self, name, type_hint=types.BASE_ALL):
        self._var_types[name] = type_hint
        return ast.Field(name), types.VARIABLE

    def method_chain(self, node):
        """Expand a chain of methods into function calls."""
        rv = None
        for prev_node, function_node in zip(node.children[:-1], node.children[1:]):
            rv = self.function_call(function_node, prev_node, rv)

        return rv

    def value(self, node):
        """Check if a value is signed."""
        value, value_hint = self.visit(node.children[1])
        if not types.check_types(types.NUMBER, value_hint):
            raise self._type_error(node, "Sign applied to non-numeric value", types.NUMBER, value_hint)

        if node.children[0] == "-":
            if isinstance(value, ast.Number):
                value.value = -value.value
            else:
                value = ast.MathOperation(ast.Number(0), "-", value)

        return value, value_hint

    def name(self, node):
        """Check for illegal use of keyword."""
        text = node["NAME"].value
        if text in keywords:
            raise self._error(node, "Invalid use of keyword", cls=EqlSyntaxError)

        return text

    def number(self, node):
        """Parse a number with a sign."""
        token = node.children[-1]
        if token.type == "UNSIGNED_INTEGER":
            value = int(token)
        else:
            value = float(token)

            if int(value) == value:
                value = int(value)

        if node["SIGN"] == "-":
            return -value
        return value

    def string(self, node):
        value = node.children[0]

        # If a 'raw' string is detected, then only unescape the quote character
        if value.startswith('?'):
            quote_char = value[1]
            return value[2:-1].replace("\\" + quote_char, quote_char)
        else:
            return ast.String.unescape(value[1:-1])

    def base_field(self, node):
        """Get a base field."""
        name = node["name"]
        text = name["NAME"]

        if text in RESERVED:
            value = RESERVED[text]
            return value, types.literal(value.type_hint)

        # validate against the remaining keywords
        name = self.visit(name)

        if name in self.preprocessor.constants:
            constant = self.preprocessor.constants[name]
            return constant.value, types.literal(constant.value.type_hint)

        # Check if it's part of the current preprocessor that we are building
        # and if it is, then return it unexpanded but with a type hint
        if name in self.new_preprocessor.constants:
            constant = self.new_preprocessor.constants[name]
            return ast.Field(name), types.literal(constant.value.type_hint)

        field = ast.Field(name)
        return self._get_field_hint(node, field)

    def field(self, node):
        """Callback function to walk the AST."""
        full_path = [s.strip() for s in re.split(r"[.\[\]]+", node.children[0])]
        full_path = [int(s) if s.isdigit() else s for s in full_path if s]

        if any(p in keywords for p in full_path):
            raise self._error(node, "Invalid use of keyword", cls=EqlSyntaxError)

        base, path = full_path[0], full_path[1:]

        # if get_variable:
        #     if base_name in RESERVED or node.sub_fields:
        #         raise self._type_error(node, "Expected {expected_type} not {field} to function", types.VARIABLE)
        #     elif base_name in self._var_types:
        #         raise self._error(node, "Reuse of variable {base}")

        #     # This can be overridden by the parent function that is parsing it
        #     return self._add_variable(node.base)
        field = ast.Field(base, path)
        return self._get_field_hint(node, field)

    def comparison(self, node):
        """Callback function to walk the AST."""
        (left, left_type), comp_op, (right, right_type) = self.visit_children(node)

        op = "==" if comp_op.type == 'EQUALS' else comp_op.value

        accepted_types = types.union(types.PRIMITIVES, types.NULL)
        error_message = "Unable to compare {expected_type} to {actual_type}"

        if not types.check_types(left_type, right_type) or \
                not types.check_types(accepted_types, left_type) or \
                not types.check_types(accepted_types, right_type):
            # check if the types can actually be compared, and don't allow comparison of nested types
            raise self._type_error(node, error_message, types.clear(left_type), types.clear(right_type))

        if op in (ast.Comparison.LT, ast.Comparison.LE, ast.Comparison.GE, ast.Comparison.GE):
            # check that <, <=, >, >= are only supported for strings or integers
            lt = types.get_type(left_type)
            rt = types.get_type(right_type)

            # string to string or number to number
            if not ((types.check_full_hint(types.STRING, lt) and types.check_full_hint(types.STRING, rt)) or
                    (types.check_full_hint(types.NUMBER, lt) and types.check_full_hint(types.NUMBER, rt))):
                raise self._type_error(node, error_message, types.clear(left_type), types.clear(right_type))

        comp_node = ast.Comparison(left, op, right)
        hint = types.get_specifier(types.union(left_type, right_type)), types.get_type(types.BOOLEAN)

        # there is no special comparison operator for wildcards, just look for * in the string
        if isinstance(right, ast.String) and '*' in right.value:
            func_call = ast.FunctionCall('wildcard', [left, right])

            if op == ast.Comparison.EQ:
                return func_call, hint
            elif op == ast.Comparison.NE:
                return ~ func_call, hint

        return comp_node, hint

    def mathop(self, node):
        """Callback function to walk the AST."""
        output, output_type = self.visit(node.children[0])

        def update_type(error_node, new_op, new_type):
            if not types.check_types(types.NUMBER, new_type):
                raise self._type_error(error_node, "Unable to {func} {actual_type}",
                                       types.NUMBER, new_type, func=ast.MathOperation.func_lookup[new_op])

            output_spec = types.get_specifier(output_type)

            if types.check_types(types.NULL, types.union(output_type, new_type)):
                return output_spec, types.union_types(types.BASE_NULL, types.BASE_NUMBER)
            else:
                return output_spec, types.BASE_NUMBER

        # update the type hint to strip non numeric information
        output_type = update_type(node.children[0], node.children[1], output_type)

        for op_token, current_node in zip(node.children[1::2], node.children[2::2]):
            op = op_token.value
            right, current_hint = self.visit(current_node)
            output_type = update_type(current_node, op, current_hint)

            # determine if this could have a null in it from a divide by 0
            if op in "%/" and (not isinstance(right, ast.Literal) or right.value == 0):
                current_hint = types.union(current_hint, types.NULL)

            output = ast.MathOperation(output, op, right)
            output_type = update_type(current_node, op, current_hint)

            if isinstance(output, ast.Literal):
                output_type = types.get_specifier(output), output.type_hint

        return output, output_type

    sum_expr = mathop
    mul_expr = mathop

    def bool_expr(self, node, cls):
        """Method for both and, or expressions."""
        terms, hints = self.unzip_hints(self.visit_children(node))

        if not self._implied_booleans:
            for lark_node, hint in zip(node.child_trees, hints):
                if not types.check_types(types.BOOLEAN, hint):
                    raise self._type_error(lark_node, "Expected {expected_type}, not {actual_type}",
                                           types.BOOLEAN, hint)

        term = cls(terms)
        return term, types.union(*hints)

    def and_expr(self, node):
        """Callback function to walk the AST."""
        return self.bool_expr(node, ast.And)

    def or_expr(self, node):
        """Callback function to walk the AST."""
        return self.bool_expr(node, ast.Or)

    def not_expr(self, node):
        """Callback function to walk the AST."""
        term, hint = self.visit(node.children[-1])

        if not self._implied_booleans:
            if not types.check_types(types.BOOLEAN, hint):
                raise self._type_error(node.child_trees[-1], "Expected {expected_type}, not {actual_type}",
                                       types.BOOLEAN, hint)

        if len(node.get_list("NOT_OP")) % 2 == 1:
            term = ~ term
            hint = types.get_specifier(hint), types.BASE_BOOLEAN

        return term, hint

    def not_in_set(self, node):
        """Method for converting `x not in (...)`."""
        rv, rv_hint = self.in_set(node)
        return ~rv, rv_hint

    def in_set(self, node):
        """Callback function to walk the AST."""
        (expr, outer_hint), (container, sub_hints) = self.visit_children(node)
        outer_spec = types.get_specifier(outer_hint)
        outer_type = types.get_type(outer_hint)
        container_specifiers = [types.get_specifier(h) for h in sub_hints]
        container_types = [types.get_type(h) for h in sub_hints]

        # Check that everything inside the container has the same type as outside
        error_message = "Unable to compare {expected_type} to {actual_type}"
        for container_node, node_type in zip(node["expressions"].children, container_types):
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
            signature = self._function_lookup.get(ast_node.name)
            if signature:
                type_hint = signature.return_value

        if any(isinstance(n, ast.Field) for n in ast_node):
            type_hint = types.dynamic(type_hint)

        return type_hint

    def function_call(self, node, prev_node=None, prev_arg=None):
        """Callback function to walk the AST."""
        function_name = self.visit(node["name"])
        argument_nodes = []
        args = []
        hints = []

        # if the base is chained from a previous function call, use that node
        if prev_arg:
            base_arg, base_hint = prev_arg
            args.append(base_arg)
            hints.append(base_hint)

        if prev_node:
            argument_nodes.append(prev_node)

        if "expressions" in node:
            argument_nodes.extend(node["expressions"].children)

        if function_name in self.preprocessor.macros:
            if prev_node and not prev_arg:
                arg, hint = self.visit(prev_node)
                args.append(arg)
                hints.append(hint)

            if node["expressions"]:
                args[len(args):], hints[len(hints):] = self.visit(node["expressions"])

            macro = self.preprocessor.macros[function_name]
            expanded = macro.expand(args)
            type_hint = self._get_type_hint(node, expanded)
            return expanded, type_hint

        elif function_name in self.new_preprocessor.macros:
            if prev_node and not prev_arg:
                arg, hint = self.visit(prev_node)
                args.append(arg)
                hints.append(hint)

            if node["expressions"]:
                args[len(args):], hints[len(hints):] = self.visit(node["expressions"])

            macro = self.new_preprocessor.macros[function_name]
            expanded = macro.expand(args)
            type_hint = self._get_type_hint(node, expanded)
            return expanded, type_hint

        signature = self._function_lookup.get(function_name)

        if signature:
            # Check for any variables in the signature, and handle their type hints differently
            variables = set(idx for idx, hint in enumerate(signature.argument_types) if hint == types.VARIABLE)

            arguments = []

            # Back up the current variable type hints for when this function goes out of scope
            old_variables = self._var_types.copy()

            # Get all of the arguments first, because they may depend on others
            # and we need to pull out all of the variables

            for idx, arg_node in enumerate(argument_nodes):
                if idx in variables:
                    if arg_node.data == "base_field":
                        variable_name = self.visit(arg_node["name"])
                        self._add_variable(variable_name)
                        arguments.append((ast.Field(variable_name), types.VARIABLE))
                    else:
                        raise self._type_error(arg_node, "Invalid argument to {name}. Expected {expected_type}",
                                               types.VARIABLE, name=function_name)
                elif idx == 0 and prev_arg:
                    arguments.append(prev_arg)
                else:
                    arguments.append(self.visit(arg_node))

            # Then validate this against the signature
            args, hints = self.unzip_hints(arguments)

            # In theory, we could do another round of validation for generics, but we'll just assume
            # that loop variables can take any shape they need to, as long as the other arguments match

            # Validate that the arguments match the function signature by type and length
            args, hints = self.validate_signature(node, signature, argument_nodes, args, hints)

            # Restore old variables, since ours are out of scope now
            self._var_types = old_variables

            # Get return value and specifier, and mark as dynamic if any of the inputs are
            output_hint = signature.return_value

            if hints and types.is_dynamic(types.union(*hints)):
                output_hint = types.dynamic(output_hint)

            return ast.FunctionCall(function_name, args, as_method=prev_node is not None), output_hint

        elif self._check_functions:
            raise self._error(node["name"], "Unknown function {NAME}")
        else:
            args = []

            if node["expressions"]:
                args, _ = self.visit(node["expressions"])

            func_node = ast.FunctionCall(function_name, args, as_method=prev_node is not None)
            return func_node, types.dynamic(types.EXPRESSION)

    # queries
    def event_query(self, node):
        """Callback function to walk the AST."""
        if node["name"] is None:
            event_type = EVENT_TYPE_ANY
            if not self.implied_any:
                raise self._error(node, "Missing event type and 'where' condition", cls=EqlSyntaxError)
        else:
            event_type = self.visit(node["name"])

            if self._schema and not self._schema.validate_event_type(event_type):
                raise self._error(node["name"], "Invalid event type: {NAME}", cls=EqlSchemaError, width=len(event_type))

        with self.scoped(event_type=event_type, query_condition=True):
            expr, hint = self.visit(node.children[-1])
            if not self._implied_booleans and not types.check_types(types.BOOLEAN, hint):
                raise self._type_error(node.children[-1], "Expected {expected_type} not {actual_type}",
                                       types.BOOLEAN, hint)

        return ast.EventQuery(event_type, expr)

    def pipe(self, node):
        """Callback function to walk the AST."""
        if not self._pipes_enabled:
            raise self._error(node, "Pipes not supported")

        pipe_name = self.visit(node["name"])
        pipe_cls = ast.PipeCommand.lookup.get(pipe_name)
        if pipe_cls is None or pipe_name not in self._allowed_pipes:
            raise self._error(node["name"], "Unknown pipe {NAME}")

        args = []
        hints = []
        arg_nodes = []

        if node["expressions"]:
            arg_nodes = node["expressions"].children
            args, hints = self.visit(node["expressions"])
        elif len(node.children) > 1:
            arg_nodes = node.children[1:]
            args, hints = self.unzip_hints(self.visit(c) for c in node.children[1:])

        args, hints = self.validate_signature(node, pipe_cls, arg_nodes, args, hints)
        self._pipe_schemas = pipe_cls.output_schemas(args, hints, self._pipe_schemas)
        return pipe_cls(args)

    def base_query(self, node):
        """Visit a sequence, join or event query."""
        return self.visit(node.children[0])

    def piped_query(self, node):
        """Callback function to walk the AST."""
        if "base_query" in node:
            first = self.visit(node["base_query"])
        elif self.implied_base:
            first = ast.EventQuery(EVENT_TYPE_ANY, ast.Boolean(True))
        else:
            raise self._error(node, "Missing base query")

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

        return ast.PipedQuery(first, self.visit_children(node["pipes"]))

    def expressions(self, node):
        """Convert a list of expressions."""
        expressions = self.visit_children(node)
        # Split out return types and the hints
        return self.unzip_hints(expressions)

    def named_subquery(self, node):
        """Callback function to walk the AST."""
        name = self.visit(node["name"])

        if not self._subqueries_enabled:
            raise self._error(node, "Subqueries not supported")
        elif self._in_pipes:
            raise self._error(node, "Not supported within pipe")
        elif name not in ast.NamedSubquery.supported_types:
            raise self._error(node["name"], "Unknown subquery type '{NAME} of'")

        query = self.visit(node["subquery"]["event_query"])
        return ast.NamedSubquery(name, query), types.dynamic(types.BOOLEAN)

    def named_params(self, node, get_param=None, position=None, close=None):
        """Callback function to walk the AST."""
        if node is None:
            return ast.NamedParams({})

        params = OrderedDict()
        if get_param is None and len(node.children) > 0:
            raise self._error(node.children[0], "Unexpected parameters")

        for param in node.children:
            key, value = get_param(param, position=position, close=close)
            if key in params:
                raise self._error(param, "Repeated parameter {name}")
            params[key] = value
        return ast.NamedParams(params)

    def subquery_by(self, node, num_values=None, position=None, close=None, get_param=None):
        """Callback function to walk the AST."""
        if not self._subqueries_enabled:
            raise self._error(node, "Subqueries not supported")

        actual_num = len(node["join_values"]["expressions"].children) if node["join_values"] else 0
        if num_values is not None and num_values != actual_num:
            if actual_num == 0:
                error_node = node
                end = True
            else:
                end = False
                error_node = node["join_values"]["expressions"].children[max(num_values, actual_num) - 1]
            message = "Expected {num} value"
            if num_values != 1:
                message += "s"
            raise self._error(error_node, message, num=num_values, end=end)

        if node["named_params"]:
            params = self.named_params(node["named_params"],
                                       get_param=get_param, position=position, close=close)
        else:
            params = ast.NamedParams()

        query = self.visit(node["subquery"]["event_query"])

        if node["join_values"]:
            with self.scoped(event_type=query.event_type):
                join_values, join_hints = self.visit(node["join_values"])
        else:
            join_values, join_hints = [], []
        return ast.SubqueryBy(query, params, join_values), join_hints

    def join_values(self, node):
        """Return all of the expressions."""
        return self.visit(node["expressions"])

    def join(self, node):
        """Callback function to walk the AST."""
        queries, close = self._get_subqueries_and_close(node)
        return ast.Join(queries, close)

    def _get_subqueries_and_close(self, node, get_param=None):
        """Helper function used by join and sequence to avoid duplicate code."""
        if not self._subqueries_enabled:
            # Raise the error earlier (instead of waiting until subquery_by) so that it's more meaningful
            raise self._error(node, "Subqueries not supported")

        # Figure out how many fields are joined by in the first query, and match across all
        subquery_nodes = node.get_list("subquery_by")
        first, first_hints = self.subquery_by(subquery_nodes[0], get_param=get_param, position=0)
        num_values = len(first.join_values)
        queries = [(first, first_hints)]

        for pos, query in enumerate(subquery_nodes[1:], 1):
            queries.append(self.subquery_by(query, num_values=num_values, get_param=get_param, position=pos))

        shared = node['join_values']
        close = None

        # Validate that each field has matching types
        default_hint = types.get_type(types.union(types.PRIMITIVES, types.NULL))
        strict_hints = [default_hint] * num_values

        if shared:
            strict_hints += [default_hint] * len(shared["expressions"].children)

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
            query_node = subquery_nodes[qpos]

            if "join_values" in query_node:
                curr_join_nodes = query_node["join_values"]["expressions"].children
            else:
                curr_join_nodes = []

            if shared:
                with self.scoped(event_type=query.query.event_type):
                    curr_shared_by, curr_shared_hints = self.visit(shared)

                curr_by_hints = curr_shared_hints + curr_by_hints
                query.join_values = curr_shared_by + query.join_values
                curr_join_nodes = shared['expressions'].children + curr_join_nodes

            # Now that they've all been built out, start to intersect the types
            for fpos, (n, h) in enumerate(zip(curr_join_nodes, curr_by_hints)):
                check_by_field(fpos, n, h)

            # Add all of the fields to the beginning of this subquery's BY fields and preserve the order
            query.join_values = unshared_fields + query.join_values

        until_node = node["until_subquery_by"]

        if until_node:
            close, close_hints = self.subquery_by(until_node["subquery_by"],
                                                  num_values=num_values, get_param=get_param, close=True)
            close_nodes = [until_node["subquery_by"]]

            if shared:
                with self.scoped(event_type=close.query.event_type):
                    shared_by, shared_hints = self.visit(shared)

                close_hints = close_hints + shared_hints
                close.join_values = shared_by + close.join_values
                close_nodes = shared['expressions'].children + close_nodes

            # Check the types of the by field
            for fpos, (n, h) in enumerate(zip(close_nodes, close_hints)):
                check_by_field(fpos, n, h)

        # Unzip the queries from the (query, hint) tuples
        queries, _ = self.unzip_hints(queries)
        return list(queries), close

    def get_sequence_parameter(self, node, **kwargs):
        """Validate that sequence parameters are working."""
        key = self.visit(node["name"])

        if len(node.children) > 1:
            value, _ = self.visit(node.children[-1])
        else:
            value = ast.Boolean(True)

        if key != 'maxspan':
            raise self._error(node, "Unknown sequence parameter {}".format(key))

        value = ast.TimeRange.convert(value)

        if not value or value.delta < datetime.timedelta(0):
            error_node = node["time_range"] or node["atom"] or node
            raise self._error(error_node, "Invalid value for {}".format(key))

        return key, value

    def get_sequence_term_parameter(self, param_node, position, close):
        """Validate that sequence parameters are working for items in sequence."""
        if not position or close:
            raise self._error(param_node, "Unexpected parameters")

        # set the default type to a literal 'true'
        value, type_hint = ast.Boolean(True), types.literal(types.BOOLEAN)
        key = self.visit(param_node["name"])

        if len(param_node.children) > 1:
            value, type_hint = self.visit(param_node.children[-1])

        if key == 'fork':
            if not types.check_types(types.literal((types.NUMBER, types.BOOLEAN)), type_hint):
                raise self._type_error(param_node,
                                       "Expected type {expected_type} value for {k}",
                                       types.literal(types.BOOLEAN))

            if value.value not in (True, False, 0, 1):
                raise self._error(param_node, "Invalid value for {k}")

        else:
            raise self._error(param_node['name'], "Unknown parameter {NAME}")

        return key, ast.Boolean(bool(value.value))

    def sequence(self, node):
        """Callback function to walk the AST."""
        if not self._subqueries_enabled:
            raise self._error(node, "Subqueries not supported")

        params = None

        if node['named_params']:
            params = self.named_params(node['named_params'], get_param=self.get_sequence_parameter)

        queries, close = self._get_subqueries_and_close(node, get_param=self.get_sequence_term_parameter)
        return ast.Sequence(queries, params, close)

    def definitions(self, node):
        """Parse all definitions."""
        return self.visit_children(node)

    # definitions
    def macro(self, node):
        """Callback function to walk the AST."""
        definition = ast.Macro(self.visit(node.children[0]),
                               self.visit(node.children[1:-1]),
                               self.visit(node.children[-1])[0])
        self.new_preprocessor.add_definition(definition)
        return definition

    def constant(self, node):
        """Callback function to walk the AST."""
        name = self.visit(node["name"])
        value, _ = self.visit(node["literal"])
        definition = ast.Constant(name, value)
        self.new_preprocessor.add_definition(definition)
        return definition


lark_parser = Lark(get_etc_file('eql.g'), debug=False,
                   propagate_positions=True, tree_class=KvTree, parser='lalr',
                   start=['piped_query', 'definition', 'definitions',
                          'query_with_definitions', 'expr', 'signed_single_atom'])


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
    if not text.strip():
        raise EqlSyntaxError("No text specified", 0, 0, text)

    # Convert everything to unicode
    text = to_unicode(text)
    if not text.endswith("\n"):
        text += "\n"

    with ParserConfig(implied_any=implied_any, implied_base=implied_base, allow_subqueries=subqueries,
                      preprocessor=preprocessor, allow_pipes=pipes) as cfg:

        load_extensions(force=False)
        exc = None
        walker = LarkToEQL(text)

        try:
            tree = lark_parser.parse(text, start=start)
        except LarkError as e:
            # Remove the original exception from the traceback
            exc = EqlSyntaxError("Invalid syntax", e.line - 1, e.column - 1, '\n'.join(walker.lines[e.line - 2:e.line]))
            if cfg.read_stack("full_traceback", debugger_attached):
                raise exc

        if exc is None:
            try:
                eql_node = walker.visit(tree)
                if not isinstance(eql_node, ast.EqlNode) and isinstance(eql_node, tuple):
                    eql_node, type_hint = eql_node

                if cfg.read_stack("optimized", True):
                    optimizer = Optimizer(recursive=True)
                    eql_node = optimizer.walk(eql_node)
                return eql_node
            except EqlError as e:
                # If full traceback mode is enabled, then re-raise the exception
                if cfg.read_stack("full_traceback", debugger_attached):
                    raise
                exc = e

    # Python 3 - avoid double exceptions if full_traceback is disabled
    raise exc


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
    return _parse(text,  "piped_query", implied_any=implied_any, implied_base=implied_base, preprocessor=preprocessor,
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
    return _parse(text, start='expr', implied_any=implied_any, preprocessor=preprocessor, subqueries=subqueries)


def parse_atom(text, cls=None):  # type: (str, type) -> ast.Field|ast.Literal
    """Parse and get an atom."""
    rule = "signed_single_atom"
    atom = _parse(text, start=rule)
    if cls is not None and not isinstance(atom, cls):
        walker = LarkToEQL(text)
        lark_tree = lark_parser.parse(text, start=rule)
        raise walker._error(lark_tree, "Expected {expected} not {actual}",
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
    return _parse(text, start='definition', preprocessor=preprocessor,
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


class TermExtractor(Interpreter, object):
    """Extract query terms from a sequence, join or flat query."""

    def __init__(self, text):
        self.text = text

    def event_query(self, tree):
        return self.text[tree.meta.start_pos:tree.meta.end_pos]

    def piped_query(self, tree):
        """Extract all terms."""
        if tree["base_query"]:
            if tree["base_query"]["event_query"]:
                return [self.visit(tree["base_query"]["event_query"])]
            return self.visit(tree["base_query"].children[0])
        return []

    def sequence(self, tree):
        """Extract the terms in the sequence."""
        return [self.visit(term["subquery"]["event_query"]) for term in tree.get_list("subquery_by")]

    # these have similar enough ASTs that this is fine for extracting terms
    join = sequence


def extract_query_terms(text, **kwargs):
    """Parse out the query terms from an event query, join or sequence.

    :param str text: EQL source text to parse
    :rtype: list[str]
    """
    # validate that it parses first so that EQL exceptions are raised
    parse_query(text, **kwargs)

    tree = lark_parser.parse(text, start="piped_query")
    extractor = TermExtractor(text)
    return list(extractor.visit(tree))
