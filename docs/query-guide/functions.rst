.. include:: ../links.rst

=========
Functions
=========
Function calls keep the core language for EQL simple but easily extendable. Functions are used to perform
math, string manipulation or more sophisticated expressions to be expressed.


.. function:: add(x, y)

  Returns ``x + y``

.. function:: arrayContains(some_array, value)

  Check if ``value`` is a member of the array ``some_array``.

  .. code-block:: eql

      // {my_array: ["value1", "value2", "value3"]}

      arrayContains(my_array, "value2")  // returns true
      arrayContains(my_array, "value4")  // returns false


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


.. function:: concat(...)

    Returns a concatenated string of all the input arguments.

    .. code-block:: eql

      concat("Process ", process_name, " executed with pid ", pid)

.. function:: divide(m, n)

    Return ``m / n``

.. function:: endsWith(x, y)

    Checks if the string ``x`` ends with the substring ``y``.


.. function:: length(s)

    Returns the length of a string. Non-string values return 0.

.. function:: modulo(m, n)

    Performs the `modulo`_ operator and returns the remainder of ``m / n``.

.. function:: multiply(x, y)

    Returns ``x * y``

.. function:: number(s[, base])

    :param: base: The `base` of a number. Default value is 10 if not provided.

    Returns a number constructed from the string ``s``.

.. function:: startsWith(x, y)

    Checks if the string ``x`` starts with the string ``y``.

.. function:: string(val)

    Returns the string representation of the value ``val``.

.. function:: stringContains(a, b)

    Returns true if ``b`` is a substring of ``a``

.. function:: subtract(x, y)

    Returns ``x - y``

.. function:: wildcard(value, wildcard, [, ... ])

    Compare a value to a list of wildcards. Returns true if any of them match.
    For example, the following two expressions are equivalent.

    .. code-block:: eql

        command_line == "* create *" or command_line == "* config *" or command_line == "* start *"

        wildcard(command_line, "* create *", "* config *", "* start *")