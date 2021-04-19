.. include:: ../links.rst

=========
Functions
=========
Function calls keep the core language for EQL simple but easily extendable. Functions are used to perform
math, string manipulation or more sophisticated expressions to be expressed.


.. function:: add(x, y)

    Returns ``x + y``

    .. versionchanged:: 0.8
        Added ``+`` operator directly.

.. function:: arrayContains(some_array, value [, ...])

    Check if ``value`` is a member of the array ``some_array``.

    .. versionchanged:: 0.7
        Support for additional arguments.

    .. code-block:: eql

        // {my_array: ["value1", "value2", "value3"]}

        arrayContains(my_array, "value2")           // returns true
        arrayContains(my_array, "value4")           // returns false
        arrayContains(my_array, "value3", "value4)  // returns true

.. function:: arrayCount(array, variable, expression)

    Count the number of matches in an array to an expression.

    .. versionadded:: 0.7

    .. code-block:: eql

        // {my_array: [{user: "root", props: [{level: 1}, {level: 2}]},
        //             {user: "guest", props: [{level: 1}]}]

        arrayCount(my_array, item, item.user == "root")                           // returns 1
        arrayCount(my_array, item, item.props[0].level == 1)                      // returns 2
        arrayCount(my_array, item, item.props[1].level == 4)                      // returns 0
        arrayCount(my_array, item, arrayCount(item.props, p, p.level == 2) == 1)  // returns 1

.. function:: arraySearch(array, variable, expression)

    Check if any member in the array matches an expression.
    Unlike :func:`arrayContains`, this can search over nested structures in arrays, and supports
    searching over arrays within arrays.

    .. code-block:: eql

        // {my_array: [{user: "root", props: [{level: 1}, {level: 2}]},
        //             {user: "guest", props: [{level: 1}]}]

        arraySearch(my_array, item, item.user == "root")                       // returns true
        arraySearch(my_array, item, item.props[0].level == 1)                  // returns true
        arraySearch(my_array, item, item.props[1].level == 4)                  // returns false
        arraySearch(my_array, item, arraySearch(item.props, p, p.level == 2))  // returns true

.. function:: between(source, left, right [, greedy=false])

    Extracts a substring from ``source`` that's also between ``left`` and ``right``.

    :param greedy: Matches the longest string when set, similar to ``.*`` vs ``.*?``.

    .. versionchanged:: 0.9.1
        Removed ``case_sensitive`` parameter

    .. code-block:: eql

        between("welcome to event query language", " ", " ")            // returns "to"
        between("welcome to event query language", " ", " ", true)      // returns "to event query"


.. function:: cidrMatch(ip_address, cidr_block [, ...])

    Returns ``true`` if the source address matches any of the provided CIDR blocks.

    .. versionadded:: 0.8

    .. code-block:: eql

        // ip_address = "192.168.152.12"
        cidrMatch(ip_address, "10.0.0.0/8", "192.168.0.0/16")     // returns true

.. function:: concat(...)

    Returns a concatenated string of all the input arguments.

    .. code-block:: eql

        concat("Process ", process_name, " executed with pid ", pid)

.. function:: divide(m, n)

    Return ``m / n``

    .. versionchanged:: 0.8
        Added ``/`` operator directly.

.. function:: endsWith(x, y)

    Checks if the string ``x`` ends with the substring ``y``.


.. function:: indexOf(source, substring [, start=0])

    Find the first position (zero-indexed) of a string where a substring is found.
    If ``start`` is provided, then this will find the first occurrence at or after the start position.

    .. code-block:: eql

        indexOf("some-subdomain.another-subdomain.com", ".")     // returns 14
        indexOf("some-subdomain.another-subdomain.com", ".", 14) // returns 14
        indexOf("some-subdomain.another-subdomain.com", ".", 15) // returns 32


.. function:: length(s)

    Returns the length of a string or array.

.. function:: match(source, pattern [, ...])

    Checks if multiple regular expressions are matched against a source string.

    .. code-block:: eql

        match("event query language", ?"[a-z]+ [a-z]+ [a-z]")   // returns true

.. function:: modulo(m, n)

    Performs the `modulo`_ operator and returns the remainder of ``m / n``.

    .. versionchanged:: 0.8
        Added ``%`` operator directly.

.. function:: multiply(x, y)

    Returns ``x * y``

    .. versionchanged:: 0.8
        Added ``*`` operator directly.

.. function:: number(s [, base=10])

    :param number base: The `base`_ of a number.

    Returns a number constructed from the string ``s``.

    .. code-block:: eql

        number("1337")                  // returns 1337
        number("0xdeadbeef", 16)        // 3735928559

.. function:: startsWith(x, y)

    Checks if the string ``x`` starts with the string ``y``.

.. function:: string(val)

    Returns the string representation of the value ``val``.

.. function:: stringContains(a, b)

    Returns true if ``b`` is a substring of ``a``

.. function:: substring(source, start [, end])

    Extracts a substring from another string between ``start`` and ``end``.
    Like other EQL functions, ``start`` and ``end`` are zero-indexed positions in the string.
    Behavior is similar to Python's `string slicing`_ (``source[start:end]``), and negative offsets are supported.

    .. code-block:: eql

        substring("event query language", 0, 5)                     // returns "event"
        substring("event query language", 0, length("event"))       // returns "event"
        substring("event query language", 6, 11)                    // returns "query"
        substring("event query language", -8)                       // returns "language"
        substring("event query language", -length("language"))      // returns "language"
        substring("event query language", -5, -1))                  // returns "guag"

.. function:: subtract(x, y)

    Returns ``x - y``

.. function:: wildcard(value, wildcard [, ... ])

    Compare a value to a list of wildcards. Returns true if any of them match.
    For example, the following two expressions are equivalent.

    .. code-block:: eql

        command_line == "* create *" or command_line == "* config *" or command_line == "* start *"

        wildcard(command_line, "* create *", "* config *", "* start *")

Methods
-------
Calling functions with values returned from other functions can often be difficult to read
for complex expressions. EQL also provides an alternative method syntax that flows more
naturally from left to right.

For instance, the expression:

.. code-block:: eql

    length(between(command_line, "-enc ", " ")) > 500

is equivalent to the method syntax:

.. code-block:: eql

    command_line:between("-enc ", " "):length() > 500
