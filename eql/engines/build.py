"""Entry point to project."""
from __future__ import print_function

from eql.ast import EqlAnalytic, PipedQuery
from eql.engines.native import PythonEngine
from eql.engines.base import TextEngine
from eql.parser import parse_analytic, parse_query
from eql.utils import is_string


def render_engine(analytics, engine_type, config=None, analytics_only=False):
    """Render engine.

    :param list[EqlAnalytic] analytics: Analytics to add to the engine
    :param dict config: List of extra files to add to the engine
    :param str engine_type: the target file extension to build
    :param boolean analytics_only: Render the analytics without the core engine code.
    :return str: Returns the base engine
    """
    engine_cls = TextEngine.extensions[engine_type]

    engine = engine_cls(config)
    engine.add_analytics(analytics)
    return engine.render(analytics_only=analytics_only)


def render_analytics(analytic_infos, engine_type, analytics_only=False, config=None):
    """Render a full text engine for multiple analytics.

    :param list[EqlAnalytic] analytic_infos: A list of parsed analytics
    :param str engine_type: The target file extension
    :param analytics_only: Render the converted analytics without including the EQL core
    :param dict config: An optional engine configuration
    """
    return render_engine(analytic_infos, engine_type, config=config, analytics_only=analytics_only)


def render_analytic(analytic, engine_type, analytics_only=False, config=None):
    """Render a full script for an EQL analytic.

    :param dict|EqlAnalytic analytic: The analytic object in AST or dictionary form
    :param str engine_type: The target file extension
    :param analytics_only: Render the converted analytics without including the EQL core
    :param dict config: An optional engine configuration
    """
    if not isinstance(analytic, EqlAnalytic):
        analytic = parse_analytic(analytic)
    return render_analytics([analytic], engine_type, config=config, analytics_only=analytics_only)


def render_query(query, engine_type, config=None):
    """Render the full script for an EQL query.

    :param str|PipedQuery query: The query text or parsed query
    :param str engine_type: The target scripting engine
    :param dict config: The configuration for PythonEngine
    """
    metadata = {}
    if not isinstance(query, PipedQuery):
        metadata['_source'] = query
        query = parse_query(query)

    analytic = EqlAnalytic(query=query, metadata=metadata)
    rendered = render_analytic(analytic, engine_type=engine_type, config=config, analytics_only=False)
    return rendered


def get_reducer(query, config=None):
    """Get a reducer to aggregate results from distributed EQL queries.

    :param str|dict|EqlAnalytic|PipedQuery query: The query text or parsed query
    :param dict config: The configuration for PythonEngine
    """
    if isinstance(query, dict):
        query = parse_analytic(query)
    elif is_string(query):
        query = parse_query(query, implied_base=True, implied_any=True)

    def reducer(inputs):
        results = []
        engine = PythonEngine(config)
        engine.add_reducer(query)
        engine.add_output_hook(results.append)

        engine.reduce_events(inputs, finalize=True)
        return results

    return reducer


def get_engine(query, config=None):
    """Run an EQL query or analytic over a list of events and get the results.

    :param str|dict|EqlAnalytic|PipedQuery query: The query text or parsed query
    :param dict config: The configuration for PythonEngine
    """
    if isinstance(query, dict):
        query = parse_analytic(query)
    elif is_string(query):
        query = parse_query(query, implied_base=True, implied_any=True)

    def run_engine(inputs):
        results = []
        engine = PythonEngine(config)
        if isinstance(query, PipedQuery):
            engine.add_query(query)
        else:
            engine.add_analytic(query)
        engine.add_output_hook(results.append)
        engine.stream_events(inputs, finalize=True)
        return results

    return run_engine


def get_post_processor(query, config=None, query_multiple=True):
    """Run an EQL query or analytic over a list of events and get the results.

    :param str|PipedQuery query: The query text or parsed query
    :param dict config: The configuration for PythonEngine
    :param bool query_multiple: Query over multiple events instead of just the first event
    """
    if not isinstance(query, PipedQuery):
        query = parse_query(query, implied_base=True, implied_any=True)

    def run_engine(inputs):
        results = []
        engine = PythonEngine(config)
        engine.add_post_processor(query, query_multiple=query_multiple)
        engine.add_output_hook(results.append)
        engine.reduce_events(inputs, finalize=True)
        return results

    return run_engine
