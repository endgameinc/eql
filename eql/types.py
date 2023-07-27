"""EQL type system."""
from enum import Enum


class TypeHint(Enum):
    """Enum of all primitive types supported by EQL."""

    Array = "array"
    Boolean = "boolean"
    Numeric = "number"
    Null = "null"
    Object = "object"
    String = "string"
    Unknown = "mixed"
    Variable = "variable"

    @classmethod
    def primitives(cls):
        """Get all primitive types."""
        return TypeHint.Boolean, TypeHint.Numeric, TypeHint.Null, TypeHint.String

    def is_primitive(self):
        """Check if a type is a primitive."""
        return self in self.primitives()

    def require_literal(self):
        """Match the type hint, but additionally require an unfolded literal."""
        return TypeFoldCheck(self, True)

    def require_dynamic(self):
        """Match the type hint, and require that it doesn't fold to a literal."""
        return TypeFoldCheck(self, False)


class TypeFoldCheck(object):
    """Type check with additional folding constraints."""

    def __init__(self, type_info, require_literal):
        """Capture node information for an AST.

        :param TypeHint type_info: The type information.
        :param bool require_literal: True if the expected value must be literal
        """
        self.type_info = type_info
        self.require_literal = require_literal

    def __eq__(self, other):
        """Compare two TypeFoldCheck."""
        return type(self) == type(other) and vars(self) == vars(other)

    def __ne__(self, other):
        """Compare two TypeFoldCheck."""
        return not (self == other)


class NodeInfo(object):
    """Class for holding full type and schema information for an expression."""

    def __init__(self, node, type_info=None, nullable=True, schema=None, source=None):
        """Capture node information for an AST.

        :param ast.EqlNode node: The EQL node in the tree.
        :param TypeHint type_info: The type information.
        :param bool nullable: Whether the current type can be compared to null
        :param object|list schema: Schema information for composite values.
        :param object source: Parse tree information for node.
        """
        self.node = node
        self.type_info = type_info or TypeHint.Unknown
        self.nullable = nullable or type_info in (TypeHint.Null, TypeHint.Unknown)
        self.schema = schema
        self.source = source

    def validate(self, expected):
        """Validate the type and the literal/dynamic requirements."""
        if expected is None:
            return False

        if isinstance(expected, tuple):
            return any(self.validate(o) for o in expected)

        return self.validate_type(expected) and self.validate_literal(expected)

    def validate_type(self, expected):
        """Validate that a type hint matches its requirement.

        :param TypeFoldCheck|TypeHint|NodeInfo expected:
        """
        if expected is None:
            return False
        elif isinstance(expected, tuple):
            return any(self.validate_type(o) for o in expected)
        elif isinstance(expected, NodeInfo):
            other_type = expected.type_info
            other_nullable = expected.nullable
        elif isinstance(expected, TypeHint):
            other_type = expected
            other_nullable = True
        elif isinstance(expected, TypeFoldCheck):
            other_type = expected.type_info
            other_nullable = True
        else:
            raise TypeError("Expected one of {}, got {}", (NodeInfo, TypeHint, TypeFoldCheck), expected)

        if self.type_info == TypeHint.Null:
            return other_nullable

        if other_type == TypeHint.Null:
            return self.nullable

        return self.type_info == other_type or TypeHint.Unknown in (self.type_info, other_type)

    def validate_literal(self, expected):
        """Check if a node matches the literal/dynamic requirements."""
        from .ast import Literal

        if expected is None:
            return True
        elif isinstance(expected, TypeFoldCheck):
            if expected.require_literal:
                return isinstance(self.node, Literal)
            else:
                return not isinstance(self.node.optimize(recursive=True), Literal)

        return True

    def __eq__(self, other):
        """Compare two NodeInfo."""
        return type(self) == type(other) and vars(self) == vars(other)

    def __ne__(self, other):
        """Compare two NodeInfo."""
        return not (self == other)
