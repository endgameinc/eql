"""Mixin for adding signature validation."""
from .types import NodeInfo  # noqa: F401


class SignatureMixin(object):
    """Type validation for arguments."""

    minimum_args = None
    argument_types = []
    additional_types = None

    @classmethod
    def validate(cls, arguments):  # type: (list[NodeInfo]) -> int
        """Find the first invalid argument. Return None if all are valid."""
        minimum_args = cls.minimum_args if cls.minimum_args is not None else len(cls.argument_types)

        if minimum_args is not None and len(arguments) < minimum_args:
            return len(arguments)

        for i, argument in enumerate(arguments):
            expected = cls.additional_types if i >= len(cls.argument_types) else cls.argument_types[i]
            status = argument.validate(expected)

            if not status:
                return i
