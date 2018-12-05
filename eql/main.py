"""EQL command line utility."""
from __future__ import print_function

import argparse
import glob
import os
import sys

from eql.engines.build import render_engine
from eql.engines.native import PythonEngine
from eql.errors import EqlError
from eql.loader import load_analytics, save_analytics
from eql.parser import parse_query
from eql.schema import use_schema
from eql.utils import load_dump, stream_stdin_events, stream_file_events


def build(args):
    """Convert an EQL engine with analytics to a target language."""
    config = load_dump(args.config) if args.config else {}

    _, ext = os.path.splitext(args.output_file)
    ext = ext[len(os.extsep):]

    with use_schema(config.get('schema')):
        if '*' in args.input_file:
            analytics = []
            for input_file in glob.glob(args.input_file):
                analytics.extend(load_analytics(input_file))
        else:
            analytics = load_analytics(args.input_file)

    if ext in ('yml', 'yaml', 'json'):
        save_analytics(analytics, args.output_file)
    else:
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
    try:
        eql_query = parse_query(args.query, implied_any=True, implied_base=True)
        engine.add_query(eql_query)
    except EqlError as e:
        print(e, file=sys.stderr)
        sys.exit(2)

    engine.stream_events(stream, finalize=False)
    engine.finalize()


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
    build_parser.add_argument('--config', help='Engine configuration')
    build_parser.add_argument('--analytics-only', action='store_true', help='Skips core engine when building target')

    query_parser = subparsers.add_parser('query', help='Query an EQL engine in a target language')
    query_parser.set_defaults(func=query)
    query_parser.add_argument('query', help='The EQL query to run over the log file')
    query_parser.add_argument('--file', '-f', help='Target file(s) to query with EQL')
    query_parser.add_argument('--encoding', '-e', help='Encoding of input file', default="utf8")
    query_parser.add_argument('--format', help='', choices=['json', 'jsonl', 'json.gz', 'jsonl.gz'])
    query_parser.add_argument('--config', help='Engine configuration')

    parsed = parser.parse_args(args)

    # this won't necessarily be set in python3
    if hasattr(parsed, 'func'):
        parsed.func(parsed)
