"""Base functionality for testing."""
import json
import os
import unittest

from eql.parser import parse_analytic
from eql.engines.base import Event

DIR = os.path.dirname(os.path.abspath(__file__))


class TestEngine(unittest.TestCase):
    """Base test with helpful methods for getting example data and queries."""

    QUERIES_FILE = os.path.join(DIR, "test_queries.json")
    EVENTS_FILE = os.path.join(DIR, "test_data.json")
    engine_name = 'base'

    _query_cache = {}

    @classmethod
    def get_analytic(cls, query_text):
        """Get a cached EQL analytic."""
        if query_text not in cls._query_cache:
            analytic_info = {
                'metadata': {'id': 'query-{:d}'.format(len(cls._query_cache)),
                             'name': query_text,
                             'analytic_version': '1.0.0'},
                'query': query_text
            }
            cls._query_cache[query_text] = parse_analytic(analytic_info)
        return cls._query_cache[query_text]

    def test_valid_analytics(self):
        """Confirm that the analytics in JSON are valid."""
        self.get_example_queries()

    @classmethod
    def get_example_queries(cls):
        """Get example queries with their expected outputs."""
        with open(cls.QUERIES_FILE, "r") as f:
            queries = json.load(f)
            for q in queries:
                analytic = cls.get_analytic(q['query'])
                analytic.metadata['_info'] = q.copy()
                q['analytic'] = analytic
            return [q for q in queries if cls.engine_name not in q.get('skip', [])]

    def validate_results(self, actual, expected, query=None):
        """Validate that a list of results matches."""
        self.assertListEqual(actual, expected,
                             "Got {} but expected {} for analytic {}".format(actual, expected, query))

    _events = None

    @classmethod
    def get_events(cls):
        """Get output events from test_data.json."""
        if cls._events is None:

            with open(cls.EVENTS_FILE, "r") as f:
                data = json.load(f)
            cls._events = [Event.from_data(d) for d in data]
        return cls._events
