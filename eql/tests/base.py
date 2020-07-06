"""Helper class for validating EQL transpilers."""
import json
import os
import unittest

import toml

from eql.parser import parse_analytic
from eql.events import Event
from eql.etc import get_etc_path
from eql.schema import EMPTY_SCHEMA


DIR = os.path.dirname(os.path.abspath(__file__))
QUERIES_FILE = get_etc_path("test_queries.toml")
EVENTS_FILE = get_etc_path("test_data.json")


class TestEngine(unittest.TestCase):
    """Base test with helpful methods for getting example data and queries."""

    engine_name = 'base'
    query_cache = {}
    schema = EMPTY_SCHEMA
    queries_file = QUERIES_FILE
    events_file = EVENTS_FILE
    __events = None

    @classmethod
    def get_analytic(cls, query_text, is_case_sensitive=None):
        """Get a cached EQL analytic."""
        cache_key = (query_text, is_case_sensitive)
        with cls.schema:
            if cache_key not in cls.query_cache:
                analytic_info = {
                    'metadata': {'id': 'query-{:d}'.format(len(cls.query_cache)), 'name': query_text},
                    'query': query_text
                }
                cls.query_cache[cache_key] = parse_analytic(analytic_info)
        return cls.query_cache[cache_key]

    @classmethod
    def get_events(cls):
        """Get output events from test_data.json."""
        if cls.__events is None:
            with open(cls.events_file, "r") as f:
                data = json.load(f)
            cls.__events = [Event.from_data(d) for d in data]
        return cls.__events

    @classmethod
    def filter_queries(cls, q):
        """Helper method for filtering the test queries for subclasses."""
        return True

    @classmethod
    def get_example_queries(cls, match_case_sensitive=False):
        """Get example queries with their expected outputs."""
        with open(cls.queries_file, "r") as f:
            queries = []
            for q in toml.load(f)["queries"]:
                case_settings = []

                if "case_sensitive" not in q and "case_insensitive" not in q:
                    case_sensitive = True
                    case_insensitive = True
                else:
                    case_sensitive = q.get("case_sensitive") is True
                    case_insensitive = q.get("case_insensitive") is True

                if case_sensitive:
                    case_settings.append(True)

                if case_insensitive:
                    case_settings.append(False)

                assert len(case_settings) > 0, q

                for cs in case_settings:
                    analytic = cls.get_analytic(q['query'], cs)
                    analytic.metadata['_info'] = q.copy()
                    q['analytic'] = analytic

                    if cs == match_case_sensitive:
                        queries.append(q)

            return list(filter(cls.filter_queries, queries))

    @classmethod
    def get_example_analytics(cls, match_case_sensitive=False):
        """Get a list of example analytics from test queries."""
        return [q["analytic"] for q in cls.get_example_queries(match_case_sensitive=match_case_sensitive)]

    def validate_results(self, actual, expected, query=None):
        """Validate that a list of results matches."""
        self.assertListEqual(actual, expected,
                             "Got {} but expected {} for analytic {}".format(actual, expected, query))

    def test_valid_analytics(self):
        """Confirm that the analytics in JSON are valid."""
        self.get_example_queries()
