"""Helper classes for the EQL preprocessor."""
from collections import OrderedDict
from string import Template

from .ast import EqlNode, Field
from .utils import ParserConfig
from .walkers import RecursiveWalker

__all__ = (
    "Definition",
    "Constant",
    "BaseMacro",
    "CustomMacro",
    "Filter",
    "Macro",
    "PreProcessor",
)


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

    def expand(self, arguments):
        """Expand a macro with a set of arguments."""
        raise NotImplementedError


class CustomMacro(BaseMacro):
    """Custom macro class to use Python callbacks to transform trees."""

    def __init__(self, name, callback):
        """Python macro to allow for more dynamic or sophisticated macros.

        :param str name: The name of the macro.
        :param (list[EqlNode]) -> EqlNode callback: A callback to expand out the macro.
        """
        super(CustomMacro, self).__init__(name)
        self.callback = callback

    def expand(self, arguments):
        """Make the callback do the dirty work for expanding the AST."""
        node = self.callback(arguments)
        return node.optimize()

    @classmethod
    def from_name(cls, name):
        """Decorator to convert a function into a :class:`~CustomMacro` object."""
        def decorator(f):
            return CustomMacro(name, f)
        return decorator


class Filter(Definition):
    """Class for EQL filter shorthand: `filter x == a where ...`."""

    __slots__ = 'name', 'query'

    def __init__(self, name, query):
        """Create a named filter, as a short-hand for common queries."""
        Definition.__init__(self, name)
        self.query = query


class Macro(BaseMacro, EqlNode):
    """Class for a macro on a node, to allow for client-side expansion."""

    __slots__ = 'name', 'parameters', 'expression'
    template = Template('macro $name($parameters) $expression')
    delims = {'parameters': ', '}

    def __init__(self, name, parameters, expression):
        """Create a named macro that takes a list of arguments and returns a parameterized expression.

        :param str name: The name of the macro.
        :param list[str]: The names of the parameters.
        :param Expression expression: The parameterized expression to return.
        """
        BaseMacro.__init__(self, name)
        EqlNode.__init__(self)
        self.parameters = parameters
        self.expression = expression

    def expand(self, arguments):
        """Expand a node.

        :param list[BaseNode node] arguments: The arguments the macro is called with
        :param Walker walker: An optional syntax tree walker.
        :param bool optimize: Return an optimized copy of the AST
        :rtype: BaseNode
        """
        if len(arguments) != len(self.parameters):
            raise ValueError("Macro {} expected {} arguments but received {}".format(
                self.name, len(self.parameters), len(arguments)))

        lookup = dict(zip(self.parameters, arguments))

        def _walk_field(node):
            if node.base in lookup and not node.path:
                return lookup[node.base].optimize()
            return node

        walker = RecursiveWalker()
        walker.register_func(Field, _walk_field)
        return walker.walk(self.expression).optimize()

    def _render(self):
        expr = self.expression.render()
        if '\n' in expr or len(expr) > 40:
            expr = '\n' + self.indent(expr)
            return self.template.substitute(name=self.name, parameters=', '.join(self.parameters), expression=expr)
        return super(Macro, self)._render()


class PreProcessor(ParserConfig):
    """An EQL preprocessor stores definitions and is used for macro expansion and constants."""

    def __init__(self, definitions=None):
        """Initialize a preprocessor environment that can load definitions."""
        self.constants = OrderedDict()  # type: dict[str, Constant]
        self.macros = OrderedDict()  # type: dict[str, BaseMacro|CustomMacro|Macro]
        self.filters = OrderedDict()  # type: dict[str, Filter]

        class PreProcessorWalker(RecursiveWalker):
            """Custom walker class for this preprocessor."""

            preprocessor = self

            def _walk_field(self, node, *args, **kwargs):
                if node.base in self.preprocessor.constants and not node.path:
                        return self.preprocessor.constants[node.base].value
                return self._walk_base_node(node, *args, **kwargs)

            def _walk_function_call(self, node, *args, **kwargs):
                if node.name in self.preprocessor.macros:
                    macro = self.preprocessor.macros[node.name]
                    arguments = [self.walk(arg, *args, **kwargs) for arg in node.arguments]
                    return macro.expand(arguments)
                return self._walk_base_node(node, *args, **kwargs)

        self.walker_cls = PreProcessorWalker
        ParserConfig.__init__(self, preprocessor=self)
        self.add_definitions(definitions or [])

    def add_definitions(self, definitions):
        """Add a list of definitions."""
        for definition in definitions:
            self.add_definition(definition)

    def add_definition(self, definition):  # type: (BaseMacro|Constant) -> None
        """Add a named definition to the preprocessor."""
        name = definition.name

        if isinstance(definition, BaseMacro):
            # The macro may call into other macros so it should be expanded
            expanded_macro = self.expand(definition)
            self.macros[name] = expanded_macro
        elif isinstance(definition, Filter):
            if name in self.filters:
                raise KeyError("Filter {} already defined".format(name))
            self.filters[name] = definition
        elif isinstance(definition, Constant):
            if name in self.constants:
                raise KeyError("Constant {} already defined".format(name))
            self.constants[name] = definition

    def expand(self, root):
        """Expand the function calls that match registered macros.

        :param EqlNode root: The input node, macro, expression, etc.
        :param bool optimize: Toggle AST optimizations while expanding
        :rtype: EqlNode
        """
        if not self.constants and not self.macros:
            return root

        return self.walker_cls().walk(root)

    def copy(self):
        """Create a shallow copy of a preprocessor."""
        preprocessor = PreProcessor()
        preprocessor.constants.update(self.constants)
        preprocessor.macros.update(self.macros)
        preprocessor.filters.update(self.filters)
        return preprocessor
