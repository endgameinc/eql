.. include:: ../links.rst

======================
Implementation Details
======================
There are optimizations for ``sequence`` and ``join`` that eliminate excessive pairing of events and
enable efficient processing of a stream of events. This is different from common database relationships,
such as a `SQL Join`_, which matches every possible pairing and can potentially be costly for
event analytics.

Sequences
---------
The underlying structure of a ``sequence``
roughly resembles a `state machine`_ of events, meaning that only one pending sequence
can be in a node at a time. If the sequence uses ``by`` for matching fields, then multiple
pending sequences can exist in a given node as long as values the values matched with ``by``
are distinct. When a pending sequence matches an event, it will override any pending sequences
in the next state with identical ``by`` values.

The state changes are described for  the per-user ``sequence`` and enumeration events below.

.. code-block:: eql

    sequence by user_name
      [process where process_name == "whoami"]
      [process where process_name == "hostname"]
      [process where process_name == "ifconfig"]

.. code-block:: javascript

    {id:  1, event_type: "process", user_name: "root", process_name: "whoami"}
    {id:  2, event_type: "process", user_name: "root", process_name: "whoami"}
    {id:  3, event_type: "process", user_name: "user", process_name: "hostname"}
    {id:  4, event_type: "process", user_name: "root", process_name: "hostname"}
    {id:  5, event_type: "process", user_name: "root", process_name: "hostname"}
    {id:  6, event_type: "process", user_name: "user", process_name: "whoami"}
    {id:  7, event_type: "process", user_name: "root", process_name: "whoami"}
    {id:  8, event_type: "process", user_name: "user", process_name: "hostname"}
    {id:  9, event_type: "process", user_name: "root", process_name: "ifconfig"}
    {id: 10, event_type: "process", user_name: "user", process_name: "ifconfig"}
    {id: 11, event_type: "process", user_name: "root", process_name: "ifconfig"}

Since the sequence is separated ``by`` ``user_name``, commands executed by ``root`` and ``user``
are independently sequenced.

.. code-block:: javascript

    {id:  1, event_type: "process", user_name: "root", process_name: "whoami"}
    // sequence [1] created in root's state 1

    {id:  2, event_type: "process", user_name: "root", process_name: "whoami"}
    // sequence [2] overwrote root's state 1

    {id:  3, event_type: "process", user_name: "user", process_name: "hostname"}
    // nothing happens, because user has an empty state 1

    {id:  4, event_type: "process", user_name: "root", process_name: "hostname"}
    // sequence [2, 4] now in root's state 2
    // root's state 1 is empty

    {id:  5, event_type: "process", user_name: "root", process_name: "hostname"}
    // root's state 1 is empty, so nothing happens

    {id:  6, event_type: "process", user_name: "user", process_name: "whoami"}
    // sequence [6] created in user's state 1

    {id:  7, event_type: "process", user_name: "root", process_name: "whoami"}
    // sequence [7] created in root's state 1

    {id:  8, event_type: "process", user_name: "user", process_name: "hostname"}
    // sequence [6, 8] now in user's state 2
    // user's state 1 is now empty

    {id:  9, event_type: "process", user_name: "root", process_name: "ifconfig"}
    // sequence [2, 4, 9] completes the sequence for root
    // root still has [7] in state 1

    {id: 10, event_type: "process", user_name: "user", process_name: "ifconfig"}
    // sequence [6, 8, 10] completes the sequence for user

    {id: 11, event_type: "process", user_name: "root", process_name: "ifconfig"}
    // nothing happens because root has an empty state 2
