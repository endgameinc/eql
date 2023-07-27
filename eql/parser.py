"""Python parser functions for EQL syntax."""
from __future__ import unicode_literals

import contextlib
import datetime
import re
import sys
from collections import defaultdict

from lark import Lark, Token, Tree
from lark.exceptions import LarkError
from lark.visitors import Interpreter

from . import ast, pipes
from .errors import (EqlError, EqlSchemaError, EqlSemanticError,
                     EqlSyntaxError, EqlTypeMismatchError)
from .etc import get_etc_file
from .functions import get_function, list_functions
from .optimizer import Optimizer
from .schema import EVENT_TYPE_ANY, EVENT_TYPE_GENERIC, Schema
from .types import NodeInfo, TypeFoldCheck, TypeHint
from .utils import ParserConfig, is_string, load_extensions, to_unicode

__all__ = (
    "allow_enum_fields",
    "extract_query_terms",
    "get_preprocessor",
    "ignore_missing_fields",
    "ignore_missing_functions",
    "implied_booleans",
    "non_nullable_fields",
    "nullable_fields",
    "parse_analytic",
    "parse_analytics",
    "parse_definition",
    "parse_definitions",
    "parse_expression",
    "parse_field",
    "parse_literal",
    "parse_query",
    "skip_optimizations",
    "strict_booleans",
)


full_tracebacks = 'pydevd' in sys.modules

NON_SPACE_WS = re.compile(r"[^\S ]+")

skip_optimizations = ParserConfig(optimized=False)
ignore_missing_functions = ParserConfig(check_functions=False)
ignore_missing_fields = ParserConfig(ignore_missing_fields=False)
implied_booleans = ParserConfig(implied_booleans=True)
strict_booleans = ParserConfig(implied_booleans=False)
nullable_fields = ParserConfig(strict_fields=False)
non_nullable_fields = ParserConfig(strict_fields=True)
allow_enum_fields = ParserConfig(enable_enum=True)
elasticsearch_syntax = ParserConfig(elasticsearch_syntax=True)
elasticsearch_validate_optional_fields = ParserConfig(elasticsearch_syntax=True, validate_optional_fields=True)
elastic_endpoint_syntax = ParserConfig(elasticsearch_syntax=True, dollar_var=True, allow_alias=True)

keywords = ("and", "by", "const", "false", "in", "join", "macro",
            "not", "null", "of", "or", "sequence", "true", "until", "with", "where"
            )

RESERVED = {n.render(): n for n in [ast.Boolean(True), ast.Boolean(False), ast.Null()]}


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
        self._implied_booleans = ParserConfig.read_stack("implied_booleans", False)
        self._in_pipes = False
        self._event_types = []
        self._schema = Schema.current()
        self._ignore_missing = ParserConfig.read_stack("ignore_missing_fields", False)
        self._strict_fields = ParserConfig.read_stack("strict_fields", False)
        self._elasticsearch_syntax = ParserConfig.read_stack("elasticsearch_syntax", False)
        self._dollar_var = ParserConfig.read_stack("dollar_var", False)
        self._validate_optional_fields = ParserConfig.read_stack("validate_optional_fields", False)
        self._allow_enum = ParserConfig.read_stack("enable_enum", False)
        self._count_keys = []
        self._pipe_schemas = []
        self._var_types = dict()
        self._check_functions = ParserConfig.read_stack("check_functions", True)
        self._stacks = defaultdict(list)
        self._alias_enabled = ParserConfig.read_stack("allow_alias", False)
        self._alias_mapping = {}
        self._in_variable = False

    @property
    def lines(self):
        """Lazily split lines in the original text."""
        if self._lines is None:
            self._lines = [t.rstrip("\r\n") for t in self.text.splitlines(True)]

        return self._lines

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

        if isinstance(node, NodeInfo):
            node = node.source

        if isinstance(node, Tree):
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

    def _type_error(self, node_info, expected_type, message=None, error_node=None, **kwargs):
        """Return an exception for type mismatches."""
        kwargs.setdefault('cls', EqlTypeMismatchError)
        expected_types = expected_type if isinstance(expected_type, tuple) else (expected_type, )
        expected_names = set()
        expected_prefix = ""
        actual_prefix = ""

        error_node = error_node or node_info.source
        message = message or "Expected {expected_type} not {actual_type}"

        for ty in expected_types:
            if isinstance(ty, NodeInfo):
                expected_names.add(ty.type_info.value)
            elif isinstance(ty, TypeHint):
                expected_names.add(ty.value)
            elif isinstance(ty, TypeFoldCheck):
                expected_names.add(ty.type_info.value)

                if not node_info.validate_literal(ty):
                    if ty.require_literal:
                        expected_prefix, actual_prefix = ("literal ", "dynamic ")
                    else:
                        expected_prefix, actual_prefix = ("dynamic ", "literal ")
            else:
                raise TypeError("Unable to raise EqlTypeMismatchError from {}".format(ty))

        if node_info.validate_type(expected_type):
            expected_names = set([node_info.type_info.value])

        return self._error(error_node, message,
                           actual_type=actual_prefix + node_info.type_info.value,
                           expected_type=expected_prefix + "/".join(expected_names), **kwargs)

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

        return Interpreter.visit(self, tree)

    def validate_signature(self, node, signature, arguments):
        # type: (KvTree, type|FunctionSignature, list[NodeInfo]) -> None
        """Validate a signature against input arguments and type hints."""
        error_node = node
        node_type = 'pipe' if issubclass(signature, ast.PipeCommand) else 'function'
        name = signature.name
        bad_index = signature.validate(arguments)

        if bad_index is None:
            # no error exists, so no need to build a message
            return

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
            error_node = arguments[max_args].source
            raise self._error(error_node, message)

        elif bad_index is not None:
            if isinstance(arguments[bad_index].source, (KvTree, Token)):
                error_node = arguments[bad_index].source

            bad_argument = arguments[bad_index]
            expected_type = signature.additional_types

            if bad_index < len(signature.argument_types):
                expected_type = signature.argument_types[bad_index]

            if expected_type is not None and not bad_argument.validate(expected_type):
                raise self._type_error(bad_argument, expected_type,
                                       "Expected {expected_type} not {actual_type} to {name}", name=name)
            raise self._error(error_node, "Invalid argument to {name}", name=name)

    def start(self, node):
        """Entry point for the grammar."""
        return self.visit(node.children[0])

    # literals
    def null(self, node):
        """Callback function to walk the AST."""
        return None

    def boolean(self, node):
        """Callback function to walk the AST."""
        return node.children[0] == "true"

    def literal(self, node):
        """Callback function to walk the AST."""
        value = ast.Literal.from_python(self.visit_children(node)[0])
        return NodeInfo(value, value.type_hint, nullable=not self._strict_fields, source=node)

    def time_range(self, node):
        """Callback function to walk the AST."""
        quantity = self.visit(node["number"])  # type: int|float
        interval = self.visit(node["name"])  # type: str

        units = [v.value for v in ast.TimeUnit.get_units()]

        if isinstance(quantity, float):
            if interval is None or interval == "s":
                guess = int(round(quantity * ast.TimeUnit.Seconds.as_milliseconds()))
                time_range = ast.TimeRange(guess, ast.TimeUnit.Milliseconds)
                raise self._error(node, "Only integer values allowed for maxspan. Did you mean {time_range}?",
                                  time_range=time_range)

            # if a valid unit was specified, then filter out the less-precise units
            if interval in units and interval != ast.TimeUnit.Milliseconds.value:
                units = units[:units.index(interval)]

                raise self._error(node, "Only integer values allowed for maxspan.\n"
                                        "Try a more precise time unit: {all_units}.", all_units=", ".join(units))

            raise self._error(node, "Only integer values allowed for maxspan.", all_units=", ".join(units))

        if interval is None:
            time_range = ast.TimeRange(quantity, ast.TimeUnit.Seconds)
            raise self._error(node, "Missing time unit. Did you mean {time_range}?", time_range=time_range)

        try:
            unit = ast.TimeUnit(interval)
        except ValueError:
            raise self._error(node["name"],
                              "Unknown time unit. Recognized units are: {all_units}.",
                              interval=interval,
                              all_units=", ".join(units))

        return ast.TimeRange(quantity, unit)

    # fields
    def _update_field_info(self, node_info, optional_syntax=False):
        type_hint = None
        allow_missing = self._schema.allow_missing or (optional_syntax and not self._validate_optional_fields)
        field = node_info.node
        schema = None
        schema_hint = None
        allow_variables = self._in_variable if self._dollar_var else True

        if self._in_pipes:
            event_schema = self._pipe_schemas[0]
            event_field = field

            if self.multiple_events:
                event_index, event_field = field.query_multiple_events()
                num_events = len(self._pipe_schemas)
                if event_index >= num_events:
                    raise self._error(node_info.sub_fields[0], "Invalid index. Event array is size {num}",
                                      num=num_events)

                event_schema = self._pipe_schemas[event_index]

            # Now that we have the schema
            event_type, = event_schema.schema.keys()
            schema_hint = event_schema.get_event_type_hint(event_type, event_field.full_path)
            allow_missing = self._schema.allow_missing

        elif not self._schema:
            return NodeInfo(node_info, TypeHint.Unknown, source=node_info)

        # check if it's a variable or using an alias
        elif field.base not in self._var_types or not allow_variables:
            event_field = field

            # alias perform no field validation on what's inside the alias
            if self._alias_enabled and field.base in self._alias_mapping:
                event_field = ast.Field(field.path[0], field.path[1:])
                event_type = self._alias_mapping[field.base]
            else:
                event_type = self.scope("event_type", default=EVENT_TYPE_ANY)

            schema_hint = self._schema.get_event_type_hint(event_type, event_field.full_path)

            # Determine if the field should be converted as an enum
            # from subtype.create -> subtype == "create"
            if schema_hint is None and self._allow_enum and event_field.path and is_string(event_field.path[-1]):
                base_field = ast.Field(event_field.base, event_field.path[:-1])
                enum_value = ast.String(event_field.path[-1])
                base_hint = self._schema.get_event_type_hint(event_type, base_field.full_path)

                if base_hint is not None and base_hint[0] == TypeHint.String:
                    comparison = ast.Comparison(base_field, ast.Comparison.EQ, enum_value)
                    return NodeInfo(comparison, TypeHint.Boolean, nullable=not self._strict_fields, source=node_info)

        else:
            # Extract out types from variables and perform validation on the nested schema
            schema_hint = self._var_types[field.base]
            type_hint, schema = schema_hint

            if field.path and type_hint != TypeHint.Unknown:
                schema_hint = Schema.get_relative_path(schema, field.path)

            if schema_hint is None and not allow_missing:
                message = "Field not recognized on variable"
                raise self._error(node_info, message, cls=EqlSchemaError)

        if schema_hint is not None:
            type_hint, schema = schema_hint

        if type_hint is None and not allow_missing:
            message = "Field not recognized"
            if event_type not in (EVENT_TYPE_ANY, EVENT_TYPE_GENERIC):
                message += " for {event_type} event"
            raise self._error(node_info, message, cls=EqlSchemaError, event_type=event_type)

        # the field could be missing, so allow for null checks unless it's explicitly disabled
        type_hint = type_hint or TypeHint.Unknown
        return NodeInfo(field, type_hint, nullable=not self._strict_fields, schema=schema, source=node_info)

    def _add_variable(self, name, type_hint=TypeHint.Unknown, schema_hint=None):
        self._var_types[name] = type_hint, schema_hint
        return NodeInfo(ast.Field(name), TypeHint.Variable)

    def method_chain(self, node):
        """Expand a chain of methods into function calls."""
        rv = None
        for prev_node, function_node in zip(node.children[:-1], node.children[1:]):
            rv = self.function_call(function_node, prev_node, rv)

        rv.source = node
        return rv

    def value(self, node):
        """Check if a value is signed."""
        value = self.visit(node.children[1])  # type: NodeInfo
        if not value.validate_type(TypeHint.Numeric):
            raise self._type_error(node, TypeHint.Numeric, "Sign applied to non-numeric value")

        if node.children[0] == "-":
            if isinstance(value, ast.Number):
                value.node.value = -value.node.value
            else:
                value.node = ast.MathOperation(ast.Number(0), "-", value.node)

        value.source = node
        return value

    def method_name(self, node):
        """Check for illegal use of keyword."""
        text = node["METHOD_START"].value[1:-1]

        if text in keywords:
            raise self._error(node, "Invalid use of keyword", cls=EqlSyntaxError)

        return text

    def name(self, node):
        """Check for illegal use of keyword."""
        if isinstance(node, KvTree):
            text = node.children[0].value
        else:
            text = str(node)

        if text.rstrip("~") in keywords:
            raise self._error(node, "Invalid use of keyword", cls=EqlSyntaxError)

        return text

    def number(self, node):
        """Parse a number with a sign."""
        token = node.children[-1]
        if token.type == "UNSIGNED_INTEGER":
            value = int(token)
        else:
            value = float(token)

        if node["SIGN"] == "-":
            return -value
        return value

    def string(self, node):
        value = node.children[0]

        if self._elasticsearch_syntax:
            if value.startswith("?") or value.startswith("'"):
                raise self._error(node, 'Invalid string literal. Use " or """ based strings.', cls=EqlSyntaxError)
            elif value.startswith('"""'):
                return value[3:-3]
            else:
                return ast.String.unescape(value[1:-1])
        else:
            if value.startswith('"""'):
                raise self._error(node, 'Invalid string literal', cls=EqlSyntaxError)
            elif value.startswith('?'):
                # If a 'raw' string is detected, then only unescape the quote character
                quote_char = value[1]
                return value[2:-1].replace("\\" + quote_char, quote_char)
            else:
                return ast.String.unescape(value[1:-1])

    def base_field(self, node):
        """Get a base field."""
        child = node.children[0]
        token = child["NAME"] or child["ESCAPED_NAME"]
        name = token.value.strip("`")

        if token.type != "ESCAPED_NAME":
            if name in RESERVED:
                value = RESERVED[name]
                return NodeInfo(value, value.type_hint, source=node)

            # validate against the remaining keywords
            self.visit(child)

        if name in self.preprocessor.constants:
            constant = self.preprocessor.constants[name]
            return NodeInfo(constant.value, constant.value.type_hint, source=node)

        # Check if it's part of the current preprocessor that we are building
        # and if it is, then return it unexpanded but with a type hint
        if name in self.new_preprocessor.constants:
            constant = self.new_preprocessor.constants[name]
            return NodeInfo(ast.Field(name), constant.value.type_hint, source=node)

        return self._update_field_info(NodeInfo(ast.Field(name), source=node))

    def varpath(self, node):
        if not self._dollar_var:
            raise self._error(node, "Invalid syntax", cls=EqlSyntaxError)
        self._in_variable = True
        visited = self.visit(node.children[0])
        self._in_variable = False
        return visited

    def _field_path(self, node, allow_optional=False):
        full_path = []
        # to get around parser ambiguities, we had to create a token to mash all of the parts together
        # but we have a separate rule "field_parts" that can safely re-parse and separate out the tokens.
        # we can walk through each token, and build the field path accordingly
        text = node.children[-1]
        optional_syntax = text.startswith("?")
        if optional_syntax:
            if not allow_optional:
                raise self._error(node, "Optional fields are not supported.", cls=EqlSyntaxError, width=1)

            text = text[1:]

        for part in lark_parser.parse(text, "field_parts").children:
            if part["NAME"]:
                name = to_unicode(part["NAME"])
                full_path.append(name)

                if name in keywords or (name == "as" and self._alias_enabled):
                    raise self._error(node, "Invalid use of keyword", cls=EqlSyntaxError)
            elif part["ESCAPED_NAME"]:
                full_path.append(to_unicode(part["ESCAPED_NAME"]).strip("`"))
            elif part["UNSIGNED_INTEGER"]:
                full_path.append(int(part["UNSIGNED_INTEGER"]))
            else:
                raise self._error(node, "Unable to parse field", cls=EqlSyntaxError)

        return optional_syntax, full_path

    def field(self, node):
        """Callback function to walk the AST."""
        optional_syntax, full_path = self._field_path(node, allow_optional=self._elasticsearch_syntax)
        base, path = full_path[0], full_path[1:]

        # if get_variable:
        #     if base_name in RESERVED or node.sub_fields:
        #         raise self._type_error(node, "Expected {expected_type} not {field} to function", types.VARIABLE)
        #     elif base_name in self._var_types:
        #         raise self._error(node, "Reuse of variable {base}")

        #     # This can be overridden by the parent function that is parsing it
        #     return self._add_variable(node.base)
        field = ast.Field(base, path)
        return self._update_field_info(NodeInfo(field, source=node), optional_syntax=optional_syntax)

    def string_predicate(self, node):
        """Callback function to walk the AST."""
        predicate = node["STRING_PREDICATE"]
        if predicate == ":":
            function_name = "wildcard"
        elif predicate in ("like", "like~"):
            function_name = "wildcard"
        elif predicate in ("regex", "regex~"):
            function_name = "match"
        else:
            raise self._error(node, message="Invalid syntax", cls=EqlSyntaxError)

        if not self._elasticsearch_syntax:
            args = ", ".join(self.text[n.meta.start_pos:n.meta.end_pos] for n in node.child_trees)
            raise self._error(node, "Invalid syntax. Try: {function_name}({args})", cls=EqlSyntaxError,
                              function_name=function_name, args=args)

        children = self.visit(node.child_trees)

        for child in children:  # type: NodeInfo
            if not child.validate_type(TypeHint.String):
                raise self._type_error(child, TypeHint.String)

        return NodeInfo(ast.FunctionCall(function_name, [child.node for child in children]), TypeHint.Boolean)

    def comparison(self, node):
        """Callback function to walk the AST."""
        left, comp_op, right = self.visit_children(node)  # type: (NodeInfo, str, NodeInfo)

        op = comp_op.value

        if op == "=":
            if self._elasticsearch_syntax:
                raise self._error(node.children[1], "Invalid syntax. Compare with == instead of =", cls=EqlSyntaxError)
            op = "=="

        accepted_types = TypeHint.primitives()
        error_message = "Invalid comparison of {expected_type} to {actual_type}"

        def check_null(expr, null_side):
            # check for `expr == null` or `expr != null`
            if isinstance(null_side.node, ast.Null) and op in (ast.Comparison.NE, ast.Comparison.EQ):
                if not expr.validate_type(TypeHint.Null):
                    # check if the types can actually be compared, and don't allow comparison of nested types
                    raise self._type_error(left, TypeHint.Null, error_message, error_node=node)

                comp = ast.IsNull(expr.node) if op == ast.Comparison.EQ else ast.IsNotNull(expr.node)
                return NodeInfo(comp, TypeHint.Boolean, nullable=not self._strict_fields)

        comp = check_null(left, right) or check_null(right, left)

        if comp is not None:
            return comp

        if not left.validate_type(accepted_types) or \
                not right.validate_type(accepted_types) or \
                not right.validate_type(left):
            # check if the types can actually be compared, and don't allow comparison of nested types
            raise self._type_error(right, left, error_message, error_node=node)

        if op in (ast.Comparison.LT, ast.Comparison.LE, ast.Comparison.GE, ast.Comparison.GT):
            # check that <, <=, >, >= are only supported for strings or numbers
            if not (left.validate_type(TypeHint.Numeric) and right.validate_type(TypeHint.Numeric) or
                    left.validate_type(TypeHint.String) and right.validate_type(TypeHint.String)):

                raise self._type_error(right, left, error_message, error_node=node)

        eql_node = ast.Comparison(left.node, op, right.node)

        # there is no special comparison operator for wildcards, just look for * in the string
        if not self._elasticsearch_syntax:
            if isinstance(right.node, ast.String) and '*' in right.node.value:
                func_call = ast.FunctionCall('wildcard', [left.node, right.node])

                if op == ast.Comparison.EQ:
                    eql_node = func_call
                elif op == ast.Comparison.NE:
                    eql_node = ~func_call

            elif isinstance(left.node, ast.String) and '*' in left.node.value:
                func_call = ast.FunctionCall('wildcard', [right.node, left.node])

                if op == ast.Comparison.EQ:
                    eql_node = func_call
                elif op == ast.Comparison.NE:
                    eql_node = ~func_call

        return NodeInfo(eql_node, TypeHint.Boolean, nullable=left.nullable or right.nullable, source=node)

    def mathop(self, node):
        """Callback function to walk the AST."""
        result = self.visit(node.children[0])
        message = "Unable to {func} {actual_type}"

        def update_type(left, new_op, right):
            # type: (NodeInfo, str, NodeInfo) -> NodeInfo
            if not right.validate_type(TypeHint.Numeric):
                raise self._type_error(right, TypeHint.Numeric, message, func=ast.MathOperation.func_lookup[new_op])

            return NodeInfo(ast.MathOperation(left.node, new_op, right.node), TypeHint.Numeric,
                            nullable=left.nullable or right.nullable)

        # update the type hint to strip non numeric information

        for op_token, current_node in zip(node.children[1::2], node.children[2::2]):
            op = op_token.value
            next_node = self.visit(current_node)
            result = update_type(result, op, next_node)

        result.source = node
        return result

    sum_expr = mathop
    mul_expr = mathop

    def bool_expr(self, node, cls):
        """Method for both and, or expressions."""
        terms = self.visit_children(node)  # type: list[NodeInfo]

        if not self._implied_booleans:
            for term in terms:
                if not term.validate_type(TypeHint.Boolean):
                    raise self._type_error(term, TypeHint.Boolean)

        return NodeInfo(cls([term.node for term in terms]), TypeHint.Boolean,
                        nullable=any(t.nullable for t in terms), source=node)

    def and_expr(self, node):
        """Callback function to walk the AST."""
        return self.bool_expr(node, ast.And)

    def or_expr(self, node):
        """Callback function to walk the AST."""
        return self.bool_expr(node, ast.Or)

    def not_expr(self, node):
        """Callback function to walk the AST."""
        term = self.visit(node.children[-1])

        if not self._implied_booleans:
            if not term.validate_type(TypeHint.Boolean):
                raise self._type_error(term, TypeHint.Boolean)

        # if there are an odd number of NOTs then we negate
        if len(node.get_list("NOT_OP")) % 2 == 1:
            term.node = ast.Not(term.node)

        return term

    def not_in_set(self, node):
        """Method for converting `x not in (...)`."""
        info = self.in_set(node)
        info.node = ~info.node
        return info

    def in_set(self, node):
        """Callback function to walk the AST."""
        if not self._elasticsearch_syntax and node.get("IN") == "in~":
            raise self._error(node, message="Invalid syntax. Explicit case-insensitivity is not supported.",
                              cls=EqlSyntaxError)

        outer, container = self.visit(node.child_trees)  # type: (NodeInfo, list[NodeInfo])

        if not outer.validate_type(TypeHint.primitives()):
            # can't compare non-primitives to sets
            raise self._type_error(outer, TypeHint.primitives())

        # Check that everything inside the container has the same type as outside
        error_message = "Unable to compare {expected_type} to {actual_type}"
        for inner in container:
            if not inner.validate_type(outer):
                raise self._type_error(inner, outer, error_message)

        # This will always evaluate to true/false, so it should be a boolean
        term = ast.InSet(outer.node, [c.node for c in container])
        nullable = outer.nullable or any(c.nullable for c in container)
        return NodeInfo(term, TypeHint.Boolean, nullable=nullable, source=node)

    def _get_type_hint(self, node, ast_node):
        """Get the recommended type hint for a node when it isn't already known.

        This will likely only get called when expanding macros, until type hints are attached to AST nodes.
        """
        type_hint = TypeHint.Unknown

        if isinstance(ast_node, ast.Literal):
            type_hint = ast_node.type_hint
        elif isinstance(ast_node, (ast.Comparison, ast.InSet)):
            type_hint = TypeHint.Boolean
        elif isinstance(ast_node, ast.Field):
            type_hint = TypeHint.Unknown

            if ast_node.base not in self._var_types:
                type_hint = self._update_field_info(NodeInfo(ast_node, source=node)).type_info

        elif isinstance(ast_node, ast.FunctionCall):
            signature = self._function_lookup.get(ast_node.name)
            if signature:
                type_hint = signature.return_value

        return type_hint

    def function_call(self, node, prev_node=None, prev_arg=None):
        """Callback function to walk the AST."""
        if node.data == "method":
            name_node = node["method_name"]
            function_name = self.method_name(name_node)
        else:
            name_node = node.children[0]
            function_name = self.name(name_node)

        args = []
        nodes = []

        if function_name.endswith("~"):
            # remove the trailing ~ and use the existing AST
            function_name = function_name.rstrip("~")

            if not self._elasticsearch_syntax:
                raise self._error(node, "Invalid syntax. Explicit case-insensitivity is not supported",
                                  cls=EqlSyntaxError)

        non_method_arguments = node["expressions"].children if node["expressions"] else []

        if prev_node:
            nodes.append(prev_node)

        if prev_arg:
            args.append(prev_arg)

        if node["expressions"]:
            nodes.extend(non_method_arguments)

        if function_name in self.preprocessor.macros:
            args.extend(self.visit(non_method_arguments))
            macro = self.preprocessor.macros[function_name]
            expanded = macro.expand([arg.node for arg in args])
            type_hint = self._get_type_hint(node, expanded)
            return NodeInfo(expanded, type_hint, source=node)

        elif function_name in self.new_preprocessor.macros:
            args.extend(self.visit(non_method_arguments))
            macro = self.new_preprocessor.macros[function_name]
            expanded = macro.expand([arg.node for arg in args])
            type_hint = self._get_type_hint(node, expanded)
            return NodeInfo(expanded, type_hint, source=node)

        signature = self._function_lookup.get(function_name)

        if signature:
            # Check for any variables in the signature, and handle their type hints differently
            variables = set(idx for idx, hint in enumerate(signature.argument_types) if hint == TypeHint.Variable)

            # Back up the current variable type hints for when this function goes out of scope
            old_variables = self._var_types.copy()

            # Get all of the arguments first, because they may depend on others
            # and we need to pull out all of the variables
            var_type = None
            var_schema = None

            for idx, arg_source in enumerate(nodes):
                if idx in variables:
                    if arg_source.data == "base_field":
                        variable_name = self.visit(arg_source["name"])
                        self._add_variable(variable_name, var_type or TypeHint.Unknown, var_schema)
                        args.append(NodeInfo(ast.Field(variable_name), TypeHint.Variable, source=arg_source))
                    elif arg_source.data == "varpath" and arg_source["base_field"]:
                        variable_name = self.visit(arg_source["base_field"]["name"])
                        self._add_variable(variable_name, var_type or TypeHint.Unknown, var_schema)
                        args.append(NodeInfo(ast.Field(variable_name), TypeHint.Variable, source=arg_source))
                    else:
                        raise self._type_error(NodeInfo(None, source=arg_source), TypeHint.Variable,
                                               "Invalid argument to {name}. Expected {expected_type}",
                                               name=function_name)
                elif idx == 0 and prev_node:
                    if prev_arg is None:
                        args.append(self.visit(prev_node))
                else:
                    args.append(self.visit(arg_source))

                # Implicitly treat variables as relative to a previous array argument.
                # If not possible, it falls back to the previously permissive behavior of treating variables
                # like an Any type.
                var_type = None
                var_schema = None

                if isinstance(args[idx].schema, list) and len(args[idx].schema) == 1:
                    nested_type = Schema.convert_to_type(args[idx].schema[0])
                    if nested_type is not None:
                        var_type, var_schema = nested_type

            # Validate that the arguments match the function signature by type and length
            self.validate_signature(node, signature, args)

            # Restore old variables, since ours are out of scope now
            self._var_types = old_variables

            expr = ast.FunctionCall(function_name, [a.node for a in args], as_method=prev_node is not None)
            nullable = any(a.nullable for a in args) or signature.sometimes_null
            return NodeInfo(expr, signature.return_value, nullable=nullable, source=node)

        elif self._check_functions:
            raise self._error(name_node, "Unknown function {function_name}", function_name=function_name)
        else:
            args = []

            if node["expressions"]:
                args = self.visit(node["expressions"])

            func_node = ast.FunctionCall(function_name, [a.node for a in args], as_method=prev_node is not None)
            return NodeInfo(func_node, source=node)

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
            node_info = self.visit(node.children[-1])  # type: NodeInfo
            if not self._implied_booleans and not node_info.validate_type(TypeHint.Boolean):
                raise self._type_error(node_info, TypeHint.Boolean, "Expected {expected_type} not {actual_type}")

        return ast.EventQuery(event_type, node_info.node)

    def pipe(self, node):
        """Callback function to walk the AST."""
        if not self._pipes_enabled:
            raise self._error(node, "Pipes not supported")

        pipe_name = self.visit(node["name"])
        pipe_cls = ast.PipeCommand.lookup.get(pipe_name)
        if pipe_cls is None or pipe_name not in self._allowed_pipes:
            raise self._error(node["name"], "Unknown pipe {NAME}")

        args = []

        if node["expressions"]:
            args = self.visit(node["expressions"])
        elif len(node.children) > 1:
            args = self.visit(node.children[1:])

        self.validate_signature(node, pipe_cls, args)
        self._pipe_schemas = pipe_cls.output_schemas(args, self._pipe_schemas)
        return pipe_cls([arg.node for arg in args])

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
        return self.visit_children(node)

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
        return NodeInfo(ast.NamedSubquery(name, query), TypeHint.Boolean, source=node)

    def subquery_by(self, node, num_values=None, position=None, close=None, allow_fork=False, allow_runs=False):
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

        kwargs = {}

        repeated_sequence = node["repeated_sequence"]

        runs_count = 1
        if repeated_sequence is not None:
            runs_count = int(node["repeated_sequence"]["UNSIGNED_INTEGER"].value)

            if allow_runs is False:
                raise self._error(repeated_sequence, "Unsupported usage of repeated syntax", cls=EqlSyntaxError)

            if runs_count <= 1:
                raise self._error(repeated_sequence, "Repeated sequence runs must be greater than 1",
                                  cls=EqlSemanticError)

        fork_param = node["fork_param"]

        if fork_param is not None:
            fork_value = fork_param["boolean"]
            if allow_fork is False or position == 0 or close:
                raise self._error(fork_param, "Fork is not allowed here")

            kwargs["fork"] = self.visit(fork_value) if fork_value is not None else True

        query = self.visit(node["subquery"]["event_query"])

        if node["join_values"]:
            with self.scoped(event_type=query.event_type):
                join_values = self.visit(node["join_values"])
        else:
            join_values = []

        node_info = NodeInfo(ast.SubqueryBy(query, [v.node for v in join_values], **kwargs), source=node)

        alias = node["sequence_alias"]
        if alias is not None:
            if self._alias_enabled is False:
                raise self._error(alias, "Unsupported usage of alias syntax", cls=EqlSyntaxError)
            alias_name = alias.get('name').get('NAME').value

            # reference the subqueries by name in alias mapping
            self._alias_mapping[alias_name] = query.event_type

        return node_info, join_values, runs_count

    def join_values(self, node):
        """Return all of the expressions."""
        return self.visit(node["expressions"])

    def join(self, node):
        """Callback function to walk the AST."""
        queries, close = self._get_subqueries_and_close(node)
        return ast.Join(queries, close)

    def _get_subqueries_and_close(self, node, allow_fork=False, allow_runs=False):
        """Helper function used by join and sequence to avoid duplicate code."""
        if not self._subqueries_enabled:
            # Raise the error earlier (instead of waiting until subquery_by) so that it's more meaningful
            raise self._error(node, "Subqueries not supported")

        # Figure out how many fields are joined by in the first query, and match across all
        subquery_nodes = node.get_list("subquery_by")
        first, first_info, _ = self.subquery_by(subquery_nodes[0], allow_fork=allow_fork,
                                                position=0, allow_runs=allow_runs)

        num_values = len(first_info)
        subqueries = [(first, first_info)]

        shared = node['join_values']
        until_node = node["until_subquery_by"]
        close = None

        if until_node:
            repeated_sequence = until_node["subquery_by"]["repeated_sequence"]
            if repeated_sequence:
                raise self._error(repeated_sequence, "Unsupported usage of repeated syntax", cls=EqlSyntaxError)
            subquery_nodes.append(until_node["subquery_by"])

        for pos, subquery in enumerate(subquery_nodes[1:], 1):
            subquery, join_values, runs_count = self.subquery_by(subquery, num_values=num_values, allow_fork=allow_fork,
                                                                 position=pos, allow_runs=allow_runs)
            multiple_subqueries = [(subquery, join_values)] * runs_count
            subqueries.extend(multiple_subqueries)

        # Validate that each field has matching types
        default_hint = TypeHint.primitives()
        strict_hints = [default_hint] * num_values

        if shared:
            strict_hints += [default_hint] * len(shared["expressions"].children)

        def check_by_field(by_pos, by_node):  # type: (int, NodeInfo) -> None
            # Check that the possible values for our field that match what we currently understand about this type
            if not by_node.validate(strict_hints[by_pos]) or not by_node.validate_literal(False):
                raise self._type_error(by_node, strict_hints[by_pos], "Unable to join {expected_type} to {actual_type}")

            # Restrict the acceptable fields from what we've seen
            if by_node.type_info != TypeHint.Unknown:
                strict_hints[by_pos] = by_node.type_info

        for qpos, (subquery, by_nodes) in enumerate(subqueries):
            if shared:
                with self.scoped(event_type=subquery.node.query.event_type):
                    by_nodes = self.visit(shared) + by_nodes

            # Now that they've all been built out, start to intersect the types
            for fpos, node in enumerate(by_nodes):
                check_by_field(fpos, node)

            # Add all of the fields to the beginning of this subquery's BY fields and preserve the order
            subquery.node.join_values = [b.node for b in by_nodes]

        if until_node:
            close, _ = subqueries.pop()
            close = close.node

        return list(q.node for q, _ in subqueries), close

    def get_sequence_parameter(self, node, **kwargs):
        """Validate that sequence parameters are working."""
        key = self.visit(node["name"])

        if len(node.children) > 1:
            value = self.visit(node.children[-1])
            value = value.node
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
        value = NodeInfo(ast.Boolean(True), TypeHint.Boolean)
        key = self.visit(param_node["name"])

        if len(param_node.children) > 1:
            value = self.visit(param_node.children[-1])

        if key == 'fork':
            if not value.validate((TypeHint.Boolean.require_literal(), TypeHint.Numeric.require_literal())):
                raise self._type_error(param_node, TypeHint.Boolean, "Expected type {expected_type} value for {k}")

            if value.node.value not in (True, False, 0, 1):
                raise self._error(param_node, "Invalid value for {k}")

        else:
            raise self._error(param_node['name'], "Unknown parameter {NAME}")

        return key, ast.Boolean(bool(value.node.value))

    def sequence(self, node):
        """Callback function to walk the AST."""
        if not self._subqueries_enabled:
            raise self._error(node, "Subqueries not supported")

        params = None

        if node['with_params']:
            params = self.time_range(node['with_params']['time_range'])

        allow_runs = self._elasticsearch_syntax

        queries, close = self._get_subqueries_and_close(node, allow_fork=True, allow_runs=allow_runs)
        if len(queries) <= 1 and not self._elasticsearch_syntax:
            raise self._error(node, "Only one item in the sequence",
                              cls=EqlSemanticError if self._elasticsearch_syntax else EqlSyntaxError)
        return ast.Sequence(queries, params, close)

    def definitions(self, node):
        """Parse all definitions."""
        return self.visit_children(node)

    # definitions
    def macro(self, node):
        """Callback function to walk the AST."""
        name = self.visit(node.children[0])
        params = self.visit(node.children[1:-1])
        body = self.visit(node.children[-1])
        definition = ast.Macro(name, params, body.node)
        self.new_preprocessor.add_definition(definition)
        return definition

    def constant(self, node):
        """Callback function to walk the AST."""
        name = self.visit(node["name"])
        value = self.visit(node["literal"])
        definition = ast.Constant(name, value.node)
        self.new_preprocessor.add_definition(definition)
        return definition


lark_parser = Lark(get_etc_file('eql.g'), debug=False,
                   propagate_positions=True, tree_class=KvTree, parser='lalr',
                   start=['piped_query', 'definition', 'definitions', 'field_parts',
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
            width = len(e.token) if hasattr(e, "token") else 1
            exc = EqlSyntaxError("Invalid syntax",
                                 line=e.line - 1, column=e.column - 1, width=width,
                                 source='\n'.join(walker.lines[e.line - 2:e.line]))
            if cfg.read_stack("full_traceback", full_tracebacks):
                raise exc

        if exc is None:
            try:
                eql_node = walker.visit(tree)
                if isinstance(eql_node, NodeInfo):
                    eql_node = eql_node.node

                if cfg.read_stack("optimized", True):
                    optimizer = Optimizer(recursive=True)
                    eql_node = optimizer.walk(eql_node)
                return eql_node
            except EqlError as e:
                # If full traceback mode is enabled, then re-raise the exception
                if cfg.read_stack("full_traceback", full_tracebacks):
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
