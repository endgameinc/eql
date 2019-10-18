.. include:: ../links.rst

=============
Basic Syntax
=============

Basic queries within EQL require an event type and a matching condition.
The two are connected using the ``where`` keyword.

At the most basic level, an event query has the structure:

.. code-block:: eql

    event where condition


More specifically, an event query may resemble:

.. code-block:: eql

    process where process_name == "svchost.exe" and command_line != "* -k *"

Conditions
----------
Individual events can be matched with EQL by specifying criteria to match the fields in the event to other fields or values.
Criteria can be combined with

Boolean operators
  .. code-block:: eql

      and  or  not

Value comparisons
  .. code-block:: eql

      <  <=  ==  !=  >=  >

Wildcard matching
  .. code-block:: eql

      name == "*some*glob*match*"
      name != "*some*glob*match*"

Function calls
  .. code-block:: eql

      concat(user_domain, "\\", user_name)
      length(command_line) > 400
      add(timestamp, 300)


Lookups against static or dynamic values
  .. code-block:: eql

      user_name in ("Administrator", "SYSTEM", "NETWORK SERVICE")
      process_name in ("cmd.exe", parent_process_name)

Strings
-------
Strings are represented with single quotes ``'`` or double quotes ``"``,
with special characters escaped by a single backslash. Additionally, raw strings are
represented with a leading ``?`` character before the string, which disables escape sequences
for all characters except the quote character.

.. code-block:: eql

  "hello world"
  "hello world with 'substring'"
  'example \t of \n escaped \b characters \r etc. \f'
  ?"String with literal 'slash' \ characters included"



Event Relationships
-------------------
Relationships between events can be used for stateful tracking within the query.
If a related event exists that matches the criteria, then it is evaluated in the query as ``true``.
Relationships can be arbitrarily nested, allowing for complex behavior and state to be tracked.
Existing relationships include ``child of``, ``descendant of`` and ``event of``.

Network activity for PowerShell processes that were not spawned from explorer.exe
  .. code-block:: eql

      network where process_name == "powershell.exe" and
          not descendant of [process where process_name == "explorer.exe"]

Grandchildren of the WMI Provider Service
  .. code-block:: eql

      process where child of [process where parent_process_name == "wmiprvse.exe"]

Text file modifications by command shells with redirection
  .. code-block:: eql

      file where file_name == "*.txt" and
          event of [process where process_name == "cmd.exe" and command_line == "* > *"]

Executable file modifications by children of PowerShell
  .. code-block:: eql

      file where file_name == "*.exe" and event of [
        process where child of [process where process_name == "powershell.exe"]
      ]
