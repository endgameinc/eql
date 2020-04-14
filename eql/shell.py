#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Wrapper around the Cmd library for EQL."""
from __future__ import print_function

import cmd
import csv
import importlib
import os
import json
import re
import sys
from collections import defaultdict

from .ast import NamedSubquery
from .engine import PythonEngine
from .errors import EqlSyntaxError, EqlParseError
from .functions import list_functions
from .parser import parse_query, keywords, allow_enum_fields
from .pipes import list_pipes, CountPipe
from .schema import Schema, EVENT_TYPE_ANY, EVENT_TYPE_GENERIC
from .table import Table
from .utils import stream_file_events, load_dump, to_unicode, is_array

try:
    import prompt_toolkit
    from prompt_toolkit.formatted_text import PygmentsTokens
    from prompt_toolkit.lexers import PygmentsLexer
    from prompt_toolkit.document import Document
    from prompt_toolkit.completion import WordCompleter, PathCompleter, Completer, Completion
    from prompt_toolkit.history import FileHistory
    from prompt_toolkit import PromptSession
except ImportError:
    prompt_toolkit = None
    Completer = object
    PygmentsTokens = None

try:
    from prompt_toolkit import print_formatted_text
except ImportError:
    print_formatted_text = print

try:
    from .highlighters import EqlLexer
except ImportError:
    EqlLexer = None


try:
    import pygments
    from pygments.styles import get_style_by_name, get_all_styles
    from prompt_toolkit.styles import style_from_pygments_cls
except ImportError:
    pygments = None
    get_style_by_name = None
    get_all_styles = None
    style_from_pygments_cls = None

# Determine the installed version of readline
readline = None
readline_type = None

LIBEDIT = "libedit"
PYREADLINE = "pyreadline"
GNUREADLINE = "gnureadline"

# Determine the input function that should be used for the prompt in a python2 and python3 compatible way
try:
    input_func = raw_input
except NameError:
    input_func = input

# Determine which version of readline is installed
for module in ["readline", "gnureadline"]:
    try:
        readline = importlib.import_module(module)
        readline_doc = (getattr(readline, "__doc__", None) or "").lower()
        if PYREADLINE in sys.modules:
            readline_type = PYREADLINE
        elif LIBEDIT in readline_doc:
            readline_type = LIBEDIT
        elif "gnu" in readline_doc.lower() or module == GNUREADLINE:
            readline_type = GNUREADLINE

        break

    except Exception:
        continue

# if we found a readline, but had an error loading _the_ readline, so we'll replace it
if readline is not None and readline not in sys.modules:
    sys.modules['readline'] = readline


def callmethod(obj, method, default=None, *args, **kwargs):
    """Call a method but get a default value if it doesn't exist."""
    method = getattr(obj, method, None)
    if callable(method):
        return method(*args, **kwargs)
    return default


class ShellCompleter(Completer):
    """Completer for shell commands and EQL syntax."""

    def __init__(self, shell):  # type: (EqlShell) -> None
        """Completer for EQL shell."""
        self.shell = shell
        self.command_completer = WordCompleter(lambda: shell.completenames(""), match_middle=True)
        self.path_completer = PathCompleter(expanduser=True)

    def get_completions(self, document, complete_event):
        """Get possible completions depending on context."""
        completer = None
        complete_remaining = False
        complete_eql = False
        first_word = None

        if not self.shell.multiline:
            self.shell.prompt_session.lexer = None
            first_word = document.text and document.text.split()[0]
            if ' ' not in document.text:
                completer = self.command_completer
            elif first_word == "search":
                complete_eql = True
                self.shell.prompt_session.lexer = self.shell.tk_lexer
            elif first_word in ("input", "config", "output"):
                completer = self.path_completer
                complete_remaining = True
        else:
            self.shell.prompt_session.lexer = self.shell.tk_lexer
            complete_eql = True

        if complete_eql:
            word = document.get_word_before_cursor()
            for match in self.shell.complete_search(word, document.text, document.cursor_position,
                                                    len(document.text), contains=True):
                yield Completion(match, -len(word))
            return

        if completer:
            if complete_remaining:
                offset = len(first_word) + 1
                path_doc = Document(document.text[offset:])
                for completion in self.path_completer.get_completions(path_doc, complete_event):
                    yield Completion(completion.text, 0, display=completion.display)
            else:
                for completion in completer.get_completions(document, complete_event):
                    yield completion
        elif first_word:
            word = document.get_word_before_cursor()
            method = getattr(self.shell, "complete_" + first_word, None)
            if method:
                for match in method(word, document.text_before_cursor, document.cursor_position, len(document.text)):
                    yield Completion(match, -len(word))


class EqlShell(cmd.Cmd, object):
    """Event Query Language interactive console application."""

    # Allow dashes
    identchars = cmd.Cmd.identchars + '-'
    default_prompt = "eql> "
    continue_prompt = " ..> "
    history_file = os.path.join(os.path.expanduser("~"), '.eql')
    ansi_invisible_re = re.compile(r"(\x1b.*?[a-z])")
    nested_field_re = re.compile(r"\b([a-zA-Z][a-zA-Z0-9_]*)((\[(\d+)\]|\.([a-zA-Z][a-zA-Z0-9_]*))+|\.|\[\d*)$")

    field_split = re.compile(r"[.\[\]]+")
    doc_header = "Available commands (type help <topic>):"

    __eql_keywords = set()

    def __init__(self, *args, **kwargs):
        """EQL Shell."""
        super(EqlShell, self).__init__(*args, **kwargs)
        self.tty = callmethod(self.stdout, "isatty", False)
        self.multiline = False
        self.stop = False
        self.last_results = []
        self.columns = []
        self.input_file = None
        self.empty_count = 0
        self.prompt_session = False
        self.config = None
        self.last_display_fn = None
        self.display_fn = None
        self.last_query = None

        if prompt_toolkit and self.tty:
            self.tk_lexer = None
            if EqlLexer:
                self.tk_lexer = PygmentsLexer(EqlLexer)

            self.tk_completer = ShellCompleter(self)
            self.tk_history = FileHistory(self.history_file + ".tk")
            style_cls = None

            # switch to something more friendly
            if get_style_by_name:
                style = get_style_by_name("rrt" if sys.platform.startswith("win") else "monokai")
                if style:
                    style_cls = style_from_pygments_cls(style)

            self.default_style = style_cls
            self.prompt_session = PromptSession(style=style_cls, history=self.tk_history,
                                                completer=self.tk_completer)

    @classmethod
    def get_keywords(cls, force=False):
        """Get the EQL keywords."""
        if force or not cls.__eql_keywords:
            wordlist = set()
            updated_keywords = set(keywords)

            updated_keywords.remove("in")
            updated_keywords.add("in (")

            wordlist.update(["true", "false", "null"])

            updated_keywords.remove("with")
            wordlist.update(["with maxspan=", "fork=true"])

            wordlist.update("{}(".format(f) for f in list_functions())
            wordlist.update("| {}".format(p) for p in list_pipes())
            updated_keywords.remove("of")
            wordlist.update(["{} of [".format(k) for k in NamedSubquery.supported_types])

            wordlist.update(updated_keywords)
            cls.__eql_keywords = list(sorted(wordlist))
        return cls.__eql_keywords

    def prompt_func(self, text=None):
        """Colorize the prompt if possible."""
        if text is None:
            text = self.prompt

        # Only use prompt_toolkit when running on multiple lines
        if self.prompt_session:
            return self.prompt_session.prompt(text)
        return input_func(text)

    @property
    def prompt(self):
        """Dynamically determine the prompt based off the current state."""
        return self.continue_prompt if self.multiline else self.default_prompt

    def emptyline(self):
        """Don't automatically run that last command on duplicate ENTER."""
        return ""

    def cmdloop(self, intro=None):
        """Patch the original cmd.Cmd for better support."""
        self.preloop()
        old_completer = None
        if readline:
            old_completer = readline.get_completer()
            readline.set_completer(self.complete)
            to_parse = self.completekey + ": complete"
            if readline_type == LIBEDIT:
                to_parse = 'bind ^I rl_complete'
            readline.parse_and_bind(to_parse)

        try:
            if intro is not None:
                self.intro = intro
            if self.intro:
                print_formatted_text(intro)

            print_formatted_text("type help to view more commands")

            self.stop = False

            while not self.stop:
                try:
                    line = self.prompt_func(self.prompt)
                    if isinstance(line, bytes):
                        line = line.decode("utf-8")
                    line = line.rstrip("\r\n")
                except EOFError:
                    print_formatted_text("")
                    self.stop = True
                    line = ""
                except KeyboardInterrupt:
                    print_formatted_text()
                    print_formatted_text("KeyboardInterrupt")
                    self.multiline = False
                    continue

                line = self.precmd(line)
                self.stop = self.onecmd(line) or self.stop
                self.stop = self.postcmd(self.stop, line)
            self.postloop()
        finally:
            if readline and old_completer:
                callmethod(readline, "set_completer", None, old_completer)

    def parseline(self, line):
        """Continue parsing multiple lines when enabled."""
        if self.multiline:
            line = self.lastcmd + "\n" + line
        cmd, arg, line = super(EqlShell, self).parseline(line)

        if self.multiline and line == self.lastcmd:
            self.empty_count += 1
        else:
            self.empty_count = 0
        return cmd, arg, line

    def onecmd(self, line):
        """Wrap exception handling."""
        try:
            return super(EqlShell, self).onecmd(line)
        except EqlParseError as err:
            self.multiline = False

            if pygments and EqlLexer and self.prompt_session:
                # Recover the original text
                err_text = to_unicode(err)
                lines = err_text.splitlines()
                print_formatted_text("\n".join(lines[:2]))
                tokens = list(pygments.lex("\n".join(lines[2:]), lexer=EqlLexer()))
                print_formatted_text(PygmentsTokens(tokens), style=self.prompt_session.style)
            else:
                print_formatted_text(err)
        except Exception as err:
            self.multiline = False
            print_formatted_text(u"{}: {}".format(type(err).__name__, err))

    def complete_single_file(self, text, line, begidx, endidx):
        """Tab completion for file paths."""
        startpos = 0

        while startpos < len(line) and line[startpos] in self.identchars:
            startpos += 1

        startpos += len(line[startpos:]) - len(line[startpos:].lstrip())
        file_path = "".join(line[startpos:])

        matches = self.complete_files(file_path)
        completions = [m[begidx - startpos:] for m in matches]
        return completions

    def default(self, line):
        """Return the line."""
        print_formatted_text("Unknown command: " + line)

    def complete_files(self, text):
        """Get tab-completion options for file paths."""
        matches = []
        directory, match = os.path.split(text)
        expanded_dir = os.path.expanduser(directory or ".")
        if not os.path.exists(expanded_dir):
            return []

        for name in os.listdir(os.path.expanduser(directory or ".")):
            if name.startswith(match):
                matches.append(os.path.join(directory, name))
        return matches[:40]

    def do_input(self, file_path):
        """Point EQL to a data file for searches to be executed against."""
        # Confirm that it loads and that events are found
        if not file_path:
            print_formatted_text("Error: File path not specified")
            return

        file_path = os.path.expanduser(file_path)
        size = [0]

        def increment(event):
            size[0] += 1
            return event

        event_stream = stream_file_events(file_path)
        schema = Schema.learn(increment(event) for event in event_stream)
        print_formatted_text("Using file {:s} with {:d} events".format(file_path, size[0]))
        Schema.default(schema)
        self.input_file = file_path

    def do_schema(self, line):
        """Show the current EQL schema used to validate queries against the input file."""
        import pprint
        pprint.pprint(Schema.current().schema)

    def do_config(self, file_path):
        """Load a config file for schema checking or other engine parameters."""
        if not file_path:
            print_formatted_text("Error: File path not specified")
            return

        config = load_dump(file_path)
        if not isinstance(config, dict):
            print_formatted_text("Invalid config data")

        self.config = config
        if config.get("schema") is not None:
            schema = Schema(**self.config["schema"])
            schema.default(schema)

    # Only enable this command if prompt toolkit is found
    if prompt_toolkit and get_all_styles:
        def do_style(self, line):
            """Change the color theme used for syntax highlighting."""
            styles = set(get_all_styles())
            if "reset" in line.split():
                self.prompt_session.style = self.default_style
                return
            elif line in styles:
                pygments_style_cls = get_style_by_name(line)
                self.prompt_session.style = style_from_pygments_cls(pygments_style_cls)
                return
            elif line:
                print_formatted_text("Invalid style\n")

            # Print the list of available styles
            self.print_topics("Available styles", list(sorted(styles)), 15, 80)

        def complete_style(self, text, line, begidx, endidx):
            """"Complete pygment styles."""
            styles = list(get_all_styles())
            styles.append("reset")
            styles = [s for s in styles if text in s]
            # sort the exact matches to the top
            styles.sort(key=lambda s: (not s.startswith(text), s))
            return styles

    complete_input = complete_single_file
    complete_config = complete_single_file
    complete_output = complete_single_file

    def do_clear(self, *args):
        """Clear the terminal."""
        if sys.platform.startswith('win'):
            os.system('cls')
        else:
            sys.stdout.write('\033[2J\033[1;1H')

    def help_search(self, *args):
        """Print help text."""
        print_formatted_text(to_unicode(EqlShell.do_search.__doc__))
        print_formatted_text("\nQueries spanning multiple lines can be terminated with two newlines or a semicolon.")

    def do_search(self, search_text):
        """Run an EQL search over the input data."""
        search_lines = search_text.splitlines(False)
        self.multiline = False

        # if only "search" is typed in, then keep prompting
        if not search_text:
            self.multiline = True
            return

        try:
            with allow_enum_fields:
                parsed_query = parse_query(search_text, implied_base=True, implied_any=True, cli=True)
        except EqlSyntaxError as exc:
            # check if the query should be continued on another line
            if (exc.line + 1) == len(search_lines) and (exc.column + 1) == len(search_lines[-1]):
                if not search_text.endswith(";"):
                    self.multiline = True
                    return
            raise

        # check if the query is fully valid, but spans multiple lines
        # we want to keep prompting until we see a semicolon, or two blank lines
        if len(self.lastcmd.splitlines(True)) > 1:
            if not search_text.endswith(";") and self.empty_count < 2:
                self.multiline = True
                return

        if not self.input_file:
            print_formatted_text("Input file required. Run `input <file>`")
            return

        engine = PythonEngine(self.config)
        self.last_results = []
        count = [0]

        def callback(results):
            count[0] += 1
            for e in results.events:
                self.last_results.append(e.data)
                if self.display_fn is None:
                    if count[0] < 100:
                        engine.print_event(e)
                    elif count[0] == 100:
                        print_formatted_text("...")
            count[0] += 1

        engine.config["flatten"] = False
        engine.add_query(parsed_query)
        engine.add_output_hook(callback)
        event_stream = stream_file_events(self.input_file)
        engine.stream_events(event_stream)

        self.last_query = parsed_query

        count = len(self.last_results)

        if self.display_fn and count:
            self.display_fn(self.last_results)

        # Unconditionally show the number of results returned
        print_formatted_text("{:d} result{} found".format(count, "" if count == 1 else "s"))

    @classmethod
    def _get_schema_matches(cls, text, path, schema):
        flat_schema = Schema({"flattened": {}})

        for v in schema.schema.values():
            flat_schema = flat_schema.merge(Schema({"flattened": v}))

        # add event types
        matches = set()
        matches.update(schema.schema)

        # for now, just drop events[0] for completing the schema
        if len(path) > 2 and path[0] == "events" and isinstance(path[1], int):
            path = path[2:]

        # determine what could be completed
        if len(path) > 1:
            type_hint = flat_schema.get_event_type_hint(EVENT_TYPE_ANY, path[:-1])

            if type_hint is None:
                return []

            type_hint, schema_hint = type_hint

        else:
            # we don't know what the event type is so it could technically be any of the top level fields
            # but the schema has already been flattened going into this
            schema_hint, = flat_schema.schema.values()

        prefix = text if text.endswith(".") or text.endswith("]") else ""

        if isinstance(schema_hint, dict):
            for key, v in schema_hint.items():
                if key == path[-1]:
                    if isinstance(v, dict):
                        matches.add(key + ".")
                    elif isinstance(v, list):
                        matches.add(key + "[")
                else:
                    matches.add(key)

        return {prefix + match for match in matches}

    def complete_search(self, text, line, begidx, endidx, fields_only=False, contains=False):  # noqa: C901
        """Complete EQL keywords or known schema fields."""
        matches = set()
        nested_match = self.nested_field_re.search(line)

        # tracking event type is hard, so instead flatten all schema event types
        schema = Schema.current()  # type: Schema

        # check for completion of nested fields
        if nested_match:
            field_path = self.field_split.split(nested_match.group(0))
            field_path = [int(f) if f.isdigit() else f for f in field_path]
            matches.update(self._get_schema_matches(text, field_path, schema))

        else:
            matches.update(self._get_schema_matches(text, [text], schema))

            if not fields_only:
                matches.update(k for k in self.get_keywords() if k.startswith(text))

                # if you're in a pipe, allow completion for only pipes
                completed = line if not text else line[:-len(text)]
                if completed.rstrip().endswith("|"):
                    return list(sorted(pipe for pipe in list_pipes() if pipe.startswith(text)))

                # require keywords to have exact (not substring) matches
                matches = {m for m in matches if m.startswith(text)}

                if schema.allow_any:
                    matches.add(EVENT_TYPE_ANY)

                if schema.allow_generic:
                    matches.add(EVENT_TYPE_GENERIC)

            for event_schema in schema.schema.values():
                for k, v in event_schema.items():
                    if isinstance(v, dict):
                        matches.add(k + ".")
                    else:
                        matches.add(k)

        if text:
            if contains:
                matches = [w for w in set(matches) if text in w]
                # show the matches that start with this first
                matches.sort(key=lambda m: (not m.startswith(text), m))
                return matches
            return list(sorted(w for w in matches if w.startswith(text)))
        return []

    def _save_csv(self, path, results):
        with open(path, "w") as output_file:
            if not results:
                return

            all_fields = set()
            array_fields = defaultdict(int)

            for result in results:
                for k, v in result.items():
                    all_fields.add(k)
                    if is_array(v):
                        array_fields[k] = max(array_fields[k], len(v))

            # now build up the columns
            array_fields = {k: v for k, v in array_fields.items() if v < 3}
            all_fields = list(sorted(all_fields))

            # Start building the csv
            csv_file = csv.writer(output_file, quoting=csv.QUOTE_MINIMAL)
            header = []
            for k in all_fields:
                if k in array_fields:
                    header.extend(["{}[{}]".format(k, i) for i in range(array_fields[k])])
                else:
                    header.append(k)

            # check for python 2 compatibility
            if type(u"") != str:
                def writerow(row):
                    csv_file.writerow([cell.encode("utf8") for cell in row])
            else:
                writerow = csv_file.writerow

            writerow(header)

            for result in results:
                row = []

                for k in all_fields:
                    value = result.get(k, "")
                    if k in array_fields:
                        for i in range(array_fields[k]):
                            if is_array(value) and i < len(value):
                                row.append(value[i])
                            else:
                                row.append("")
                    else:
                        row.append(value)

                writerow([to_unicode(r) if r is not None else "" for r in row])

    def complete_display(self, text, line, *args):
        """Complete on or off values."""
        options = ("off", "on")

        if not text:
            return options
        else:
            return [o for o in options if text.startswith(o)]

    def do_display(self, line):
        """Toggle the displaying of results."""
        if line == "off":
            if self.display_fn:
                self.last_display_fn = self.display_fn or self.last_display_fn
            self.display_fn = False
        elif line == "on":
            if not self.display_fn:
                self.display_fn = self.last_display_fn
                self.last_display_fn = False
            self.display_fn = self.display_fn or None
        else:
            print("Expected on/off to display.", file=sys.stderr)

    def do_output(self, path):
        """Save the most recent results as a .json, .jsonl, or .csv file."""
        _, extension = os.path.splitext(path)
        extension = extension.lower()

        if extension == ".csv":
            self._save_csv(path, self.last_results)
        elif extension == ".jsonl":
            with open(path, "w") as f:
                for result in self.last_results:
                    f.write(json.dumps(result))
                    f.write("\n")
        elif extension == ".json":
            with open(path, "w") as f:
                json.dump(self.last_results, f, indent=2, sort_keys=True)
        else:
            print("Unknown file type: {:s}".format(extension), file=sys.stderr)
            return

        print("Saved {:d} results to {:s}".format(len(self.last_results or []), path))

    def do_shell(self, line):
        """Run a shell command."""
        os.system(line)

    def do_table(self, line):
        """Render the most recent results as a table and arguments will update the columns."""
        if not line and not self.last_results:
            print_formatted_text("No results to render, and no columns specified.")
            print_formatted_text("Try performing a `search` command")
            return

        if "--clear" in line:
            self.display_fn = None
            return

        if line or not self.display_fn:
            columns = [c.strip() for c in re.split(r"[,\s]+", line) if c.strip()]
            count_keys = {"count", "key", "percent"}

            def display_table(results):
                dynamic_columns = list(columns)
                show_counts = set(count_keys).intersection(set(columns))
                show_counts = show_counts or any(c.startswith("key[") or c.startswith("key.") for c in dynamic_columns)

                if any(isinstance(pipe, CountPipe) for pipe in self.last_query.pipes) and not show_counts:
                    last_count = next(pipe for pipe in reversed(self.last_query.pipes) if isinstance(pipe, CountPipe))
                    # Figure out how many keys there are
                    dynamic_columns = {"count": None}

                    if last_count.arguments:
                        dynamic_columns["key"] = True

                        if len(self.last_results) > 1:
                            for key in ("percent", "total_hosts"):
                                if key in self.last_results[0]:
                                    dynamic_columns[key] = None

                table = Table.from_list(dynamic_columns, results)
                for i, row in enumerate(table.lines()):
                    # , bold=(0 < i <= len(table._headers)))
                    print_formatted_text(row)

            self.display_fn = display_table

        if self.last_results and self.display_fn:
            self.display_fn(self.last_results)

    def complete_table(self, *args):
        """Tab completion for tables."""
        return self.complete_search(*args, fields_only=True, contains=True)

    def do_quit(self, line):
        """Exit the shell."""
        return True

    def do_exit(self, line):
        """Exit the shell."""
        return True
