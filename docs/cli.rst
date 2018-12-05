.. include:: links.rst

======================
Command-Line Utility
======================

The EQL python package provides a command line interface that will stream over `JSON`_,
and output as matches are found. An input file can be provided with ``-f`` in JSON or as lines of JSON (``.jsonl``).
Lines of JSON can also be processed as streams from stdin.


.. code-block:: console

    $ eql query 'process where true | head 1' -f input.json
    {"timestamp": 131485083040000000, "process_name": "System Idle Process"}

    $ eql query "process where true | head 1" < input.jsonl
    {"timestamp": 131485083040000000, "process_name": "System Idle Process"}

    $ cat input.jsonl | eql query "process where true" | head -n 1
    {"timestamp": 131485083040000000, "process_name": "System Idle Process"}

    $ eql query "process where true | count process_name | head 3" -f tmp.jsonl
    {"count": 1, "percent": 0.125, "key": "application.exe"}
    {"count": 2, "percent": 0.25, "key": "software.exe"}
    {"count": 2, "percent": 0.25, "key": "tools.exe"}

Additionally, the CLI allows for pieces of the query to be missing.
The base query ``process where true`` can be skipped altogether if pipes are present.


.. code-block:: console

    $ eql query '| head 1' -f input.jsonl
    {"timestamp": 131485083040000000, "process_name": "System Idle Process"}


Additionally, ``any where process_name == "application.exe"`` is equivalent to ``process_name == "application.exe"``

.. code-block:: console

    $ eql query "process_name == '*.exe' | count process_name | head 3" -f tmp.jsonl
    {"count": 1, "percent": 0.125, "key": "application.exe"}
    {"count": 2, "percent": 0.25, "key": "software.exe"}
    {"count": 2, "percent": 0.25, "key": "tools.exe"}


Detailed Usage
==============
.. code-block:: console

    $ eql -h
    usage: eql [-h] [--version] {build,query} ...

``eql build``
^^^^^^^^^^^^^^
.. code-block:: console

    $ eql build -h
    usage: eql build [-h] [--config CONFIG] [--analytics-only] input_file output_file

    positional arguments:
      input_file       Input analytic file(s) (.json, .yml, .toml)
      output_file      Output engine file

    optional arguments:
      --config CONFIG  Engine configuration
      --analytics-only     Skips core engine when building target

``eql query``
^^^^^^^^^^^^^^
.. code-block:: console

    $ eql query -h
    usage: eql query [-h] [--file FILE] [--encoding ENCODING]
                     [--format {json,jsonl}] [--config CONFIG]
                     query

    positional arguments:
      query                 The EQL query to run over the log file

    optional arguments:
      --file FILE, -f FILE  Target file(s) to query with EQL
      --encoding ENCODING, -e ENCODING
                            Encoding of input file (utf8, utf16, etc)
      --format {json,jsonl,json.gz,jsonl.gz}
                            File type. If not specified, defaults to the extension for --file
      --config CONFIG       Engine configuration
