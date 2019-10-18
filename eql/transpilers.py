"""Core EQL functionality for query translation."""
from .ast import PreProcessor, Field
from .parser import parse_definitions, ignore_missing_functions
from .utils import is_string, ParserConfig
from .walkers import ConfigurableWalker


class NodeMethods(dict):
    """Dictionary of methods to lookup by a key (usually a class)."""

    class UnknownNode(KeyError):
        """Unregistered node."""

    def add(self, key):
        """Add a callback method to the dictionary for a specific class.

        :param BaseNode key: The class of the object passed in
        """
        def decorator(f):
            """The function decorator that registers a method by node type."""
            assert key not in self
            self[key] = f
            return f

        return decorator

    def replace(self, key):
        """Add a callback method to the dictionary for a specific class.

        :param BaseNode key: The class of the object passed in
        """
        def decorator(f):
            """The function decorator that registers a method by node type."""
            self[key] = f
            return f

        return decorator

    def __call__(self, transpiler, node, *args, **kwargs):  # type: (BaseTranspiler, BaseNode) -> object
        """Call the bound method for a node."""
        cls = type(node)
        try:
            unbound = self[cls]
        except KeyError as e:
            raise NodeMethods.UnknownNode(e)
        return unbound(transpiler, node, *args, **kwargs)


class BaseTranspiler(ConfigurableWalker):
    """Base Transpiler class for converting ASTs from one language to another."""

    def __init__(self, config=None):
        """Instantiate the transpiler."""
        super(BaseTranspiler, self).__init__(config)
        self.config = config or {}
        self.stack = []  # type: list[BaseNode]
        self._counter = 0

    @staticmethod
    def is_variable(node):  # type: (EqlNode) -> bool
        """Check if a node is a variable for a callback function."""
        return isinstance(node, Field) and not node.path

    def counter(self, reset=False):
        """Increment counter and get current value."""
        current = 0 if reset else self._counter
        self._counter = current + 1
        return current

    def push(self, node):  # type: (BaseNode) -> None
        """Push node onto stack."""
        self.stack.append(node)

    def pop(self):  # type: () -> BaseNode
        """Pop node from stack."""
        return self.stack.pop()

    def pop_many(self, count):  # type: (int) -> list[BaseNode]
        """Pop multiple nodes from the stack."""
        popped = self.stack[-count:]
        self.stack[-count:] = []
        return popped


class BaseEngine(ConfigurableWalker, ParserConfig):
    """Add and render EQL analytics to the generic engines."""

    def __init__(self, config=None):
        """Create the engine with an optional list of files."""
        ConfigurableWalker.__init__(self, config)
        self.analytics = []  # type: list[EqlAnalytic]
        self.preprocessor = PreProcessor()

        # Set the context for `with engine:` syntax
        ParserConfig.__init__(self, preprocessor=self.preprocessor, schema=self._schema)

        with self.schema:
            definitions = self.get_config('definitions', [])
            if is_string(definitions):
                definitions = parse_definitions(definitions)

            self.preprocessor.add_definitions(definitions)

            for path in self.get_config('definitions_files', []):
                # skip missing function errors, because these are actually macros
                with ignore_missing_functions, open(path, 'r') as f:
                    definitions = parse_definitions(f.read())
                self.preprocessor.add_definitions(definitions)

    def add_analytic(self, analytic):
        # type: (EqlAnalytic) -> None
        """Add a single analytic to the engine."""
        analytic = self.preprocessor.expand(analytic)
        self.analytics.append(analytic)

    def add_analytics(self, analytics):
        """Add multiple analytics to the engine."""
        for analytic in analytics:
            self.add_analytic(analytic)


class TextEngine(BaseEngine):
    """Converter for EQL to a target language script."""

    base_files = []
    transpiler_cls = BaseTranspiler
    extensions = {}

    def __init__(self, config=None):
        """Create the engine with an optional list of files."""
        super(TextEngine, self).__init__(config)
        self.files = []
        self.files.extend(self.base_files)
        self.converted = []
        self.transpiler = self.transpiler_cls(self.config)

    def _include_file(self, path):
        with open(path, 'r') as f:
            contents = '\n'.join(line.rstrip() for line in f)
            return contents

    def add_analytic(self, analytic):
        """Convert analytic and add to engine."""
        expanded_analytic = self.preprocessor.expand(analytic)
        converted = self.transpiler.convert(expanded_analytic)
        self.analytics.append(analytic)
        self.converted.append(converted)

    def include_files(self):
        """Generate lines of text for including files."""
        output_lines = []
        for path in self.files:
            output_lines.extend(self._include_file(path).splitlines())
        return output_lines

    def render_analytics(self):
        """Render the converted analytics as lines of text."""
        output_lines = []

        for converted_analytic in self.converted:
            rendered_analytic = converted_analytic.render()
            for line in rendered_analytic.splitlines():
                output_lines.append(line.rstrip())
        return output_lines

    def render(self, analytics_only=False):
        """Create the output text for the rendered engine and analytics."""
        output_lines = []

        if not analytics_only:
            output_lines.extend(self.include_files())

        output_lines.extend(self.render_analytics())
        return '\n'.join(output_lines)


def register_extension(ext):
    """Decorator used for registering TextEngines with specific file extensions building."""
    def decorator(cls):
        TextEngine.extensions[ext] = cls
        return cls
    return decorator
