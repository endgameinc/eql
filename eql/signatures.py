"""Mixin for adding signature validation."""
from eql.types import EXPRESSION, check_full_hint


class SignatureMixin(object):
    """Type validation for arguments."""

    minimum_args = None
    # maximum_args = None
    argument_types = []
    additional_types = None

    @classmethod
    def validate(cls, arguments, type_hints=None):
        """Find the first invalid argument. Return None if all are valid."""
        minimum_args = cls.minimum_args if cls.minimum_args is not None else len(cls.argument_types)

        if minimum_args is not None and len(arguments) < minimum_args:
            return len(arguments), arguments, type_hints

        # if self.additional_types is not None and self.maximum_args is not None:
        #     if len(self.arguments) > self.maximum_args:
        #         return self.arguments[self.maximum_args or len(self.argument_types)]

        if type_hints is None:
            type_hints = [EXPRESSION] * len(arguments)

        for i, node_hint in enumerate(type_hints):
            if i >= len(cls.argument_types):
                status = check_full_hint(cls.additional_types, node_hint)
            else:
                status = check_full_hint(cls.argument_types[i], node_hint)

            if not status:
                return i, arguments, type_hints

        return None, arguments, type_hints
