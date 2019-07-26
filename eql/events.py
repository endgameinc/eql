"""Base class for constructing an analytic engine with analytics."""
from collections import namedtuple

from .schema import EVENT_TYPE_GENERIC
from .utils import is_string


class Event(namedtuple('Event', ['type', 'time', 'data'])):
    """Event for python engine in EQL."""

    @classmethod
    def from_data(cls, data):
        """Load an event from a dictionary.

        :param dict data: Dictionary with the event type, time, and keys.
        """
        data = data.get('data_buffer', data)
        timestamp = data.get('timestamp', 0)

        if is_string(data.get('event_type')):
            event_type = data['event_type']
        elif 'event_type_full' in data:
            event_type = data['event_type_full']
            if event_type.endswith('_event'):
                event_type = event_type[:-len('_event')]
        else:
            event_type = EVENT_TYPE_GENERIC

        return cls(event_type, timestamp, data)

    def copy(self):
        """Create a copy of the event."""
        data = self.data.copy()
        return Event(self.type, self.time, data)


class AnalyticOutput(namedtuple('AnalyticOutput', ['analytic_id', 'events'])):
    """AnalyticOutput for python engine in EQL."""

    @classmethod
    def from_data(cls, events, analytic_id=None):  # type: (list[dict], str) -> AnalyticOutput
        """Load up an analytic output event."""
        return cls(analytic_id, [Event.from_data(e) for e in events])
