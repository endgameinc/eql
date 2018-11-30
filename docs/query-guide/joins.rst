=====
Joins
=====
In EQL, ``join`` is used to link unordered events that may share properties. This is
similar to ``sequence``, but lacks time constraints.

Basic structure
  .. code-block:: eql

      join  // by shared_field1, shared_field2, ...
        [event_type1 where condition1] // by field1
        [event_type2 where condition2] // by field2
        ...
        [event_typeN where conditionN] // by field3


This is useful when identifying multiple connections between two network endpoints with different ports.
With ``join``, events can happen in any order, and when all events match, the ``join`` is completed.

.. code-block:: eql

    join by source_ip, destination_ip
      [network where destination_port == 3389]  // RDP
      [network where destination_port == 135]   // RPC
      [network where destination_port == 445]   // SMB

Like sequences, events can also be joined ``until`` an expiration event is met.
For instance, it may be useful to identify processes with registry, network, and file activity.

.. code-block:: eql

    join by pid
      [process where true]
      [network where true]
      [registry where true]
      [file where true]

    until [process where event_subtype_full == "termination_event"]
