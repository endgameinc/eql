"""EQL exceptions."""
import re


__all__ = (
    "EqlError",
    "ParseError",
    "SchemaError"
)


class EqlError(Exception):
    """Base class for EQL errors."""


class ParseError(EqlError):
    """EQL Parsing Error."""

    template = "Error at ({}:{}) {}:\n{}\n{}^"

    def __init__(self, error_msg, line, column, source):
        """Create error."""
        self.error_msg = error_msg
        self.line = line
        self.column = column
        self.source = source
        leading = re.sub(r'[^\t]', ' ', source)[:column]
        message = self.template.format(line + 1, column + 1, error_msg, source, leading)
        self.message = message
        super(ParseError, self).__init__(message)


class SchemaError(ParseError):
    """Error for unknown event types."""
