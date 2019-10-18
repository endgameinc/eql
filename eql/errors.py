"""EQL exceptions."""
import re


__all__ = (
    "EqlError",
    "EqlParseError",
    "EqlCompileError",
    "EqlSchemaError",
    "EqlSyntaxError",
    "EqlSemanticError",
    "EqlTypeMismatchError",
)


class EqlError(Exception):
    """Base class for EQL errors."""


class EqlCompileError(EqlError):
    """Base exception class for compiling EQL to other languages."""


class EqlParseError(EqlError):
    """EQL Parsing Error."""

    template = u"Error at line:{},column:{}\n{}\n{}\n{}"

    def __init__(self, error_msg, line, column, source, width=1, trailer=None):
        """Create error."""
        self.error_msg = error_msg
        self.line = line
        self.column = column
        self.source = source
        self.trailer = trailer
        leading = re.sub(r'[^\t]', ' ', source)[:column]
        self.caret = leading + ("^" * width)
        message = self.template.format(line + 1, column + 1, error_msg, source, self.caret)
        if trailer:
            message += "\n" + trailer

        super(EqlParseError, self).__init__(message)


class EqlSyntaxError(EqlParseError):
    """Error with EQL syntax."""


class EqlSemanticError(EqlParseError):
    """Error with EQL semantics."""


class EqlSchemaError(EqlSemanticError):
    """Error for missing fields."""


class EqlTypeMismatchError(EqlSemanticError):
    """Error when validating types."""
