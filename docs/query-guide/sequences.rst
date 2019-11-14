.. include:: ../links.rst

============
Sequences
============

Many behaviors are more complex and are best described with an ordered ``sequence`` of multiple events over a short interval.
Complex behaviors may share properties between events in the sequence or require careful handling of state.

Core sequence template
  .. code-block:: eql

    sequence
      [event_type1 where condition1]
      [event_type2 where condition2]
      ...
      [event_typeN where conditionN]

An example of simple behavior that can spans multiple events is a network logon over Remote Desktop.
With a ``maxspan`` of 30 seconds, we would expect to see an incoming network connection from a host,
followed by a separate event for the remote authentication success or failure.

.. code-block:: eql

    sequence with maxspan=30s
      [network where destination_port==3389 and event_subtype_full="*_accept_event*"]
      [security where event_id in (4624, 4625) and logon_type == 10]

Although the ``sequence`` connects the two events temporally, it doesn't prove that they are related.
There could be incoming attempts over Remote Desktop from multiple computers, leading to more
network and security events. The sequence can be constrained ``by`` matching fields,
so that the network connection and the logon event must share the same source host.

.. code-block:: eql

    sequence with maxspan=30s
      [network where destination_port==3389 and event_subtype_full="*_accept_event"] by source_address
      [security where event_id in (4624, 4625) and logon_type == 10] by ip_address

For some sequences, multiple values need to be shared across the sequence.
One example for this is a user that creates a file and shortly executes it.

.. code-block:: eql

    sequence with maxspan=5m
      [ file where file_name == "*.exe"] by user_name, file_path
      [ process where true] by user_name, process_path

Since some fields are in common across all events, this could be represented more succinctly
by moving ``by user_name`` to the top of the query.

.. code-block:: eql

    sequence by user_name with maxspan=5m
      [ file where file_name == "*.exe"] by file_path
      [ process where true] by process_path


Managing State
--------------
Occasionally, a ``sequence`` needs to carefully manage and expire state. Sequences are valid
``until`` a specific event occurs. This can help expire non-unique identifiers and reduce memory usage.

Handles and process identifiers are frequently reused. Stateful ``sequence`` tracking avoids invalid pairs of events.
Within Windows, a process identifier (PID) is only unique while a process is running, but can be reused after its termination.
When building a ``sequence`` of process identifiers, a process termination will cause all state to be invalidated and thrown away.


For instance, if ``whoami.exe`` executed from a batch file, matching ppid of ``whoami.exe`` to the pid of ``cmd.exe``
can only be done while the parent process is alive. As a result, the sequence is valid ``until``
the matching termination event occurs.

.. code-block:: eql

    sequence
      [ process where process_name == "cmd.exe" and command_line == "* *.bat*" and event_subtype_full == "creation_event"] by pid
      [ process where process_name == "whoami.exe" and event_subtype_full == "creation_event"] by ppid
    until [ process where event_subtype_full == "termination_event"] by pid
