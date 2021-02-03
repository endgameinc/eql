"""EQL command line utility."""
from __future__ import print_function

import argparse
import glob
import os
import sys

from .build import render_engine
from .engine import PythonEngine
from .errors import EqlError
from .loader import load_analytics, save_analytics
from .parser import parse_query, parse_expression
from .transpilers import TextEngine
from .utils import load_dump, stream_stdin_events, stream_file_events
from .walkers import ConfigurableWalker

BANNER = "\n".join([
    "===================",
    "     EQL SHELL     ",
    "===================",
])


def build(args):
    """Convert an EQL engine with analytics to a target language."""
    config = load_dump(args.config) if args.config else {}

    _, ext = os.path.splitext(args.output_file)
    engine_type = (args.engine_type or ext).lstrip(".")
    walker = ConfigurableWalker(config)

    with walker.schema:
        if '*' in args.input_file:
            analytics = []
            for input_file in glob.glob(args.input_file):
                analytics.extend(load_analytics(input_file))
        else:
            analytics = load_analytics(args.input_file)

    if engine_type in ('yml', 'yaml', 'json'):
        save_analytics(analytics, args.output_file)
    else:
        try:
            TextEngine.extensions[engine_type]
        except KeyError:
            print(u"Unknown extension {}".format(engine_type), file=sys.stderr)
            return 2

        output = render_engine(analytics, engine_type=ext, config=config, analytics_only=args.analytics_only)
        with open(args.output_file, "w") as f:
            f.write(output)


def query(args):
    """Query over an input file."""
    if args.file:
        stream = stream_file_events(args.file, args.format, args.encoding)
    else:
        stream = stream_stdin_events(args.format)

    config = {'print': True}
    if args.config:
        config.update(load_dump(args.config))

    engine = PythonEngine(config)

    with engine.schema:
        try:
            eql_query = parse_query(args.query, implied_any=True, implied_base=True)
            engine.add_query(eql_query)
        except EqlError as e:
            print(e, file=sys.stderr)
            sys.exit(2)

    engine.stream_events(stream, finalize=False)
    engine.finalize()


def optimize_expression(args):
    """Entry point for testing the optimizer."""
    try:
        parsed = parse_expression(args.expression)
    except EqlError as e:
        print(e)
        sys.exit(2)

    print(parsed)


def shell_main(args):
    """Entry point for the EQL shell."""
    from .shell import EqlShell
    shell = EqlShell()

    print(BANNER)

    if args.config:
        shell.do_config(args.config)

    if args.file:
        shell.do_input(args.file)

    shell.cmdloop()


def main(args=None):
    """Entry point for EQL command line utility."""
    import eql
    parser = argparse.ArgumentParser(description='Event Query Language')
    parser.add_argument('--version', '-V', action='version', version='%s %s' % (eql.__name__, eql.__version__))

    subparsers = parser.add_subparsers(help='Sub Command Help')

    build_parser = subparsers.add_parser('build', help='Build an EQL engine in a target language')
    build_parser.set_defaults(func=build)
    build_parser.add_argument('input_file', help='Input analytics file(s) (.yml or .json)')
    build_parser.add_argument('output_file', help='Output analytics engine file')
    build_parser.add_argument('--engine_type', help='Engine type. Autodetected from output extension if not provided')
    build_parser.add_argument('--analytics-only', action='store_true', help='Skips core engine when building target')

    query_parser = subparsers.add_parser('optimize', help='Optimize an EQL expression')
    query_parser.set_defaults(func=optimize_expression)
    query_parser.add_argument('expression', help='EQL expression to optimize')

    query_parser = subparsers.add_parser('query', help='Run an EQL query over stdin or a data file')
    query_parser.set_defaults(func=query)
    query_parser.add_argument('query', help='The EQL query to run over the log file')
    query_parser.add_argument('--encoding', '-e', help='Encoding of input file', default="utf8")
    query_parser.add_argument('--format', help='', choices=['json', 'jsonl', 'json.gz', 'jsonl.gz'])

    shell_parser = subparsers.add_parser('shell', help='Run an EQL query over stdin or a data file')
    shell_parser.set_defaults(func=shell_main)

    for p in (parser, build_parser, query_parser, shell_parser):
        p.add_argument('--config', '-c', help='Engine configuration')

    for p in (parser, query_parser, shell_parser):
        p.add_argument('--file', '-f', help='Target file(s) to query with EQL')

    parsed = parser.parse_args(args)

    try:
        if hasattr(parsed, 'func'):
            return parsed.func(parsed)
        else:
            return shell_main(parsed)
    except KeyboardInterrupt:
        pass
