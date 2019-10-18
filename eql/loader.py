"""Serialize analytics to and from disk."""
from .ast import EqlAnalytic  # noqa
from .parser import parse_analytic, parse_analytics
from .utils import load_dump, save_dump


def load_analytic(filename):
    """Load analytic."""
    analytic = load_dump(filename)
    return parse_analytic(analytic)


def load_analytics(filename):
    """Load analytics."""
    analytics = load_dump(filename)
    if isinstance(analytics, dict):
        analytics = analytics['analytics']  # type: list
    return parse_analytics(analytics)


def save_analytics(analytics, filename):
    # type: (list[EqlAnalytic], str) -> None
    """Save analytics."""
    rendered = [analytic.render() for analytic in analytics]

    save_dump({'analytics': rendered}, filename)


def save_analytic(analytic, filename):
    """Save analytic."""
    save_dump(analytic.render(), filename)
