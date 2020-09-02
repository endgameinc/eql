.. include:: ../links.rst

=====
Pipes
=====
Queries can include pipes for post-processing of events, and can be used for enrichment,
aggregations, statistics and filtering.


``count``
---------
The ``count`` pipe will return only statistics. If no arguments are passed, then it returns the total number of events.
Otherwise, it returns the number of occurrences for each unique value. Stats are returned in the form


Count the total number of events
  .. code-block:: eql

      process where true | count

      // results look like
      // {"count": 100, "key": totals"}


Count the number of times each value occurs
  .. code-block:: eql

    process where true | count process_name

    // results look like
    // {"count": 100, "key": "cmd.exe", "percent": 0.5}
    // {"count": 50, "key": "powershell.exe", "percent": 0.25}
    // {"count": 50, "key": "net.exe", "percent": 0.25}


Count the number of times a set of values occur
  .. code-block:: eql

    process where true | count parent_process_name, process_name

    // results look like
    // {"count": 100, "key": ["explorer.exe", "cmd.exe"], "percent": 0.5}
    // {"count": 50, "key": ["explorer.exe", "powershell.exe"], "percent": 0.25}
    // {"count": 50, "key": ["cmd.exe", "net.exe"], "percent": 0.25}


``unique``
----------
The ``unique`` pipe will only return the first matching result through the pipe. Unless a `sort`_ pipe exists before it,
events will be ordered chronologically.

Get the first matching process for each unique name
  .. code-block:: eql

      process where true | unique process_name

Get the first result for multiple of values
  .. code-block:: eql

      process where true | unique process_name, command_line

``filter``
----------
The ``filter`` pipe will only output events that match the criteria. With simple queries, this can be accomplished
by adding ``and`` to the search criteria. It's most commonly used to filter sequences or with other pipes.


Find network destinations that were first seen after May 5, 2018
  .. code-block:: eql

      network where true
      | unique destination_address, destination_port
      | filter timestamp_utc >= "2018-05-01"


Find a process with an argument `a` that wrote files to a folder in `AppData`. Use `| filter` to only match sequences where the process event contained `rar` in the process_name or the file event had a file_name that ended with `.rar`.

  .. code-block:: eql

      sequence by unique_pid
        [process where command_line == "* a *"]
        [file where file_path == "*\\AppData\\*"]
      | filter events[0].process_name == "*rar*" or events[1].file_name == "*.rar"

``unique_count``
----------------
The ``unique_count`` pipe combines the filtering of `unique`_ with the stats from `count`_. For ``unique_count``,
the original event is returned but with the fields ``count`` and ``percent`` added.

Get the first result per unique value(s), with added count information
  .. code-block:: eql

      process where true | unique_count process_name | filter count < 5

``head``
--------
The ``head`` pipe is similar to the `UNIX head`_ command and will output the first N events coming through the pipe.

Get the first fifty unique powershell commands
  .. code-block:: eql

      process where process_name == "powershell.exe"
      | unique command_line
      | head 50

``tail``
---------
The ``tail`` pipe is similar to the `UNIX tail`_ command and will output the latest events coming through the pipe.

Get the most recent ten logon events
  .. code-block:: eql

      security where event_id == 4624
      | tail 10

``sort``
---------
The ``sort`` pipe will reorder events coming through the pipe. Sorting can be done with one or multiple values.

.. warning::

    In general, ``sort`` will buffer all events coming into the pipe, and will sort them all at once.
    It's often good practice to bound the number of events into the pipe.

    For instance, the following query could be slow and require significant memory usage on a busy system.

    .. code-block:: eql

        file where true | sort file_name


Get the top five network connections that transmitted the most data
  .. code-block:: eql

    network where total_out_bytes > 100000000
    | sort total_out_bytes
    | tail 5

