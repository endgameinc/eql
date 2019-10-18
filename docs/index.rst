===============================
Event Query Language
===============================

.. image:: _static/eql.png
    :alt: eql logo
    :scale: 50%

EQL is a language that can match events, generate sequences, stack data, build aggregations, and perform analysis.
EQL is schemaless and supports multiple database backends.
It supports field lookups, boolean logic, comparisons, wildcard matching, and function calls.
EQL also has a preprocessor that can perform parse and translation time evaluation, allowing for easily sharable components between queries.


.. image:: _static/eql-whoami.jpg
    :alt: what is EQL
    :scale: 50%


Getting Started
^^^^^^^^^^^^^^^^
The EQL module current supports Python 2.7 and 3.5+. Assuming a supported Python version is installed, run the command:

.. code-block:: console

    $ pip install eql

If Python is configured and already in the PATH, then ``eql`` will be readily available, and can be checked by running the command:

.. code-block:: console

     $ eql --version
     eql 0.7.0

From there, try a :download:`sample json file <_static/example.json>` and test it with EQL.

.. code-block:: console

    $ eql query -f example.json "process where process_name == 'explorer.exe'"

    {"command_line": "C:\\Windows\\Explorer.EXE", "event_type": "process", "md5": "ac4c51eb24aa95b77f705ab159189e24", "pid": 2460, "ppid": 3052, "process_name": "explorer.exe", "process_path": "C:\\Windows\\explorer.exe", "subtype": "create", "timestamp": 131485997150000000, "user": "research\\researcher", "user_domain": "research", "user_name": "researcher"}



Next Steps
^^^^^^^^^^

- Check out the :doc:`query-guide/index` for a crash course on writing EQL queries
- View usage for the :doc:`cli`
- Explore the :doc:`api/index` for advanced usage or incorporating EQL into other projects
- Browse a `library of EQL analytics <https://eqllib.readthedocs.io>`_

.. toctree::
    :maxdepth: 1
    :caption: Contents
    :hidden:

    query-guide/index
    cli
    api/index
    resources
    licenses

License
^^^^^^^^^^
Check the :doc:`license <licenses>`
