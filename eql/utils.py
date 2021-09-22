"""Generic utility functions for analytic_engines."""
import codecs
import gzip
import io
import json
import os
import sys
import threading

PLUGIN_PREFIX = "eql_"
CASE_INSENSITIVE = True
_loaded_plugins = False

# Python2 and Python3 compatible type checking
unicode_t = type(u"")
long_t = type(int(1e100))

try:
    chr_compat = unichr
except NameError:
    chr_compat = chr

if unicode_t == str:
    strings = str,
    to_unicode = str
else:
    strings = str, unicode_t
    to_unicode = unicode_t

if long_t != int:
    numbers = (int, float, long_t)
else:
    numbers = int, float


# Optionally load dynamic loaders
try:
    import yaml
except ImportError:
    yaml = None

try:
    import toml
except ImportError:
    toml = None


def is_string(s):
    """Check if a python object is a unicode or ascii string."""
    return isinstance(s, strings)


def is_number(n):
    """Check if a python object is a unicode or ascii string."""
    return isinstance(n, numbers) and not isinstance(n, bool)


def is_array(a):
    """Check if a number is array-like."""
    return isinstance(a, (list, tuple))


def is_insensitive():
    """Check if insensitivity is enabled."""
    return CASE_INSENSITIVE


def fold_case(s):
    """Helper function for normalizing case for strings."""
    if is_insensitive() and is_string(s):
        return s.lower()
    return s


def str_presenter(dumper, data):
    """Patch YAML so that it folds the long query strings."""
    if len(data.splitlines()) > 1:  # check for multiline string
        return dumper.represent_scalar('tag:yaml.org,2002:str', data, style='|')
    return dumper.represent_scalar('tag:yaml.org,2002:str', data)


def get_type_converter(items):
    """Get a python callback function that can convert None to observed typed values."""
    items = iter(items)

    def get_empty(v):
        """Get the empty version of a value."""
        return None if v is None else type(v)()

    def default_converter(_):
        """Convert things to None when no other options available."""
        return None

    try:
        first = next(items)
    except StopIteration:
        return default_converter

    if not isinstance(first, (tuple, list)):
        empty = get_empty(first)

        if empty is None:
            for item in items:
                if item is not None:
                    empty = get_empty(item)
                    break

        return lambda x: x or empty

    else:
        empty_values = [get_empty(f) for f in first]

        for item_tuple in items:
            for i, item in enumerate(item_tuple):
                if item is not None and empty_values[i] is None:
                    # Update it with an empty value
                    empty_values[i] = type(item)()

            if all(v is not None for v in empty_values):
                break

        def convert_types(tup):
            """Take a tuple and convert each key to make sure it's not None."""
            return tuple(tup[i] or empty_i for i, empty_i in enumerate(empty_values))

        return convert_types


if yaml is not None:
    yaml.add_representer(str, str_presenter)
    if str != unicode_t:
        yaml.add_representer(unicode_t, str_presenter)


def load_dump(filename):
    """Load dump."""
    extension = filename.split('.').pop().lower()

    with open(filename) as f:
        if extension in ('yml', 'yaml'):
            assert yaml, "PyYAML module not found"
            return yaml.safe_load(f)
        elif extension == 'toml':
            assert toml, "TOML module not found"
            return toml.load(f)
        elif extension == 'json':
            return json.load(f)

    raise ValueError("Unsupported file type {}".format(extension))


def save_dump(contents, filename):
    """Save dump."""
    extension = filename.split('.').pop().lower()

    with open(filename, 'w') as f:
        if extension in ('yml', 'yaml'):
            assert yaml, "PyYAML module not found"
            yaml.dump(contents, stream=f, explicit_start=True, allow_unicode=True, default_flow_style=False, indent=2)
        elif extension == 'json':
            json.dump(contents, fp=f, indent=2, sort_keys=True)
        elif extension == 'toml':
            assert toml, "TOML module not found"
            toml.dump(contents, f)
        else:
            raise ValueError("Unsupported file type {}".format(extension))


def stream_json_lines(json_input):
    """Iterate over json lines to get Events."""
    decoder = json.JSONDecoder()
    for line in json_input:
        if "{" in line:
            yield decoder.decode(line)


def stream_file_events(file_path, file_format=None, encoding="utf8"):
    """Stream a file as JSON.

    :param str file_path: Path to the file
    :param str file_format: One of json/jsonl [.gz]
    :param str encoding: File encoding (ascii, utf8, utf16, etc.)
    """
    gz_ext = '.gz'

    if not file_format:
        base_path, file_format = os.path.splitext(file_path)
        if file_format == gz_ext:
            base_path, file_format = os.path.splitext(file_path[:-len(gz_ext)])
            file_format += gz_ext

    if file_format.endswith(gz_ext):
        file_format = file_format[:-len(gz_ext)]
        decoder = codecs.getreader(encoding)
        handle = decoder(gzip.open(file_path, 'rb'))
    else:
        handle = io.open(file_path, encoding=encoding)

    with handle:
        for event in stream_events(handle, file_format=file_format):
            yield event


def stream_stdin_events(file_format=None):
    """Stream a file as JSON.

    :param str file_format: One of json.jgz, json.gz
    """
    gz_ext = '.gz'
    file_format = file_format or 'jsonl'
    f = sys.stdin

    if file_format.endswith(gz_ext):
        file_format = file_format[:-len(gz_ext)]
        f = gzip.GzipFile(mode='r', fileobj=sys.stdin)

    for event in stream_events(f, file_format):
        yield event


def stream_events(fileobj, file_format="json"):
    """Stream events from a file handle.

    :param file fileobj: Handle to a file or stream
    :param str file_format: JSON or JSONL
    """
    file_format = file_format.lstrip(".")

    if file_format == 'jsonl':
        return stream_json_lines(fileobj)
    elif file_format == 'json':
        return json.load(fileobj)

    raise NotImplementedError("Unexpected format: {}".format(file_format))


def is_stateful(query):
    """Determine if a query requires any state tracking or if logic is atomic.

    :param ast.PipedQuery|ast.Analytic query: The parsed query AST to analyze
    :rtype: bool
    """
    # Resolve circular dependency for is_stateless
    from . import ast  # noqa: E402
    from . import pipes  # noqa: E402

    if isinstance(query, ast.EqlAnalytic):
        query = query.query

    elif not isinstance(query, ast.EqlNode):
        raise TypeError("unsupported type {} to eql.utils.is_stateful. Expected {}".format(type(query), ast.EqlNode))

    stateful_nodes = (
        ast.SubqueryBy,  # join/sequence
        ast.NamedSubquery,  # child/descendant/event of
        pipes.CountPipe, pipes.UniqueCountPipe,  # pipes count/unique_count

        # some pipe combinations, such as "| sort field | head 5" are questionable
    )

    return any(isinstance(node, stateful_nodes) for node in query)


def get_query_type(query):
    """Get the type of a query (sequence/join/event).

    :param ast.PipedQuery|ast.Analytic query: The parsed query AST to analyze
    :rtype: str
    """
    from . import ast  # noqa: E402

    if isinstance(query, ast.EqlAnalytic):
        query = query.query

    elif not isinstance(query, ast.PipedQuery):
        raise TypeError("unsupported type {} to eql.utils.get_query_type. Expected {}".format(type(query), ast.EqlNode))

    if isinstance(query.first, ast.Sequence):
        return "sequence"
    elif isinstance(query.first, ast.Join):
        return "join"
    elif isinstance(query.first, ast.EventQuery):
        return "event"
    else:
        raise TypeError("Unknown query type: {}".format(type(query.first)))


def match_kv(condition):
    """Take a list of key value pairs and generate an EQL expression.

    :param dict condition: The source text query
    :rtype: ast.Expression
    """
    # Resolve circular dependency for match_kv
    from . import ast  # noqa: E402
    from .parser import parse_field

    if not isinstance(condition, dict):
        raise TypeError("unsupported type {} to match_kv. Expected {}".format(type(condition), ast.EqlNode))

    and_node = ast.Boolean(True)

    for field_text, field_match in sorted(condition.items()):
        if not isinstance(field_match, (list, tuple)):
            field_match = [field_match]

        field_node = parse_field(field_text)

        exact = []
        wildcards = []
        for term in field_match:
            literal = ast.Literal.from_python(term)  # this may raise a TypeError
            if isinstance(literal, ast.String) and "*" in literal.value:
                wildcards.append(literal)
            else:
                exact.append(literal)

        match_node = ast.InSet(field_node, exact).optimize()
        if wildcards:
            match_node |= ast.FunctionCall("wildcard", [field_node] + wildcards)
        and_node &= match_node

    return and_node


def uses_ancestry(query):
    """Determine if a query requires process ancestry tracking.

    :param ast.PipedQuery|ast.Analytic query: The parsed query AST to analyze
    :rtype: bool
    """
    from . import ast  # noqa: E402

    if isinstance(query, ast.EqlAnalytic):
        query = query.query

    elif not isinstance(query, ast.EqlNode):
        raise TypeError("unsupported type {} to eql.utils.uses_ancestry. Expected {}".format(type(query), ast.EqlNode))

    return any(isinstance(node, ast.NamedSubquery) for node in query)


def get_required_event_types(query):
    """Get a set of all event types required for the query.

    :param ast.PipedQuery|ast.Analytic query: The parsed query AST to analyze
    :rtype: set[str]
    """
    from . import ast  # noqa: E402

    if isinstance(query, ast.EqlAnalytic):
        query = query.query

    elif not isinstance(query, ast.EqlNode):
        raise TypeError("unsupported type {} to eql.utils.uses_ancestry. Expected {}".format(type(query), ast.EqlNode))

    return set(node.event_type for node in query if isinstance(node, ast.EventQuery))


def get_output_types(query):
    """Get the output event types for a query."""
    from .walkers import RecursiveWalker
    from .ast import EqlAnalytic, PipedQuery

    if isinstance(query, EqlAnalytic):
        query = query.query

    elif not isinstance(query, PipedQuery):
        raise TypeError("unsupported type {} to get_output_types. Expected {}".format(type(query), PipedQuery))

    walker = RecursiveWalker()
    walker.walk(query)

    return walker.output_event_types


def load_extensions(force=False):
    """Load EQL extensions."""
    global _loaded_plugins

    if force or not _loaded_plugins:
        import pkgutil
        import importlib

        _loaded_plugins = True

        for module_loader, name, ispkg in pkgutil.iter_modules():
            if name.startswith(PLUGIN_PREFIX):
                importlib.import_module(name)


class ParserConfig(object):
    """Context manager for handling parser configurations."""

    __stacks = threading.local()

    def __init__(self, *managers, **config):
        """Set the current status."""
        self.managers = managers
        self.context = {k: v for k, v in config.items() if v is not None}
        super(ParserConfig, self).__init__()

    @classmethod
    def get_stack(cls, name):
        """Get a stack and initialize it to empty."""
        return cls.__stacks.__dict__.setdefault(name, [])

    @classmethod
    def push_stack(cls, name, value):
        """Push a value onto a stack for the current thread."""
        cls.get_stack(name).append(value)

    @classmethod
    def pop_stack(cls, name):
        """Pop the last value of the thread."""
        return cls.get_stack(name).pop()

    @classmethod
    def read_stack(cls, name, default=None, silent=True):
        """Read the current value of the thread."""
        stack = cls.get_stack(name)
        if silent and len(stack) == 0:
            return default
        return stack[-1]

    def __enter__(self):
        """Enter a with statement."""
        for mgr in self.managers:
            mgr.__enter__()

        for k, v in self.context.items():
            self.push_stack(k, v)

        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Pop from the stack."""
        for k in self.context:
            self.pop_stack(k)

        for mgr in reversed(self.managers):
            mgr.__exit__(exc_type, exc_val, exc_tb)
