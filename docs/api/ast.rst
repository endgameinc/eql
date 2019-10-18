====================
Abstract Syntax Tree
====================

.. automodule:: eql.ast

.. autoclass:: eql.ast.BaseNode

    .. automethod:: eql.ast.BaseNode.render

.. autoclass:: eql.ast.EqlNode

.. autoclass:: eql.ast.Walker
    :members:

.. autoclass:: eql.walkers.RecursiveWalker
.. autoclass:: eql.walkers.DepthFirstWalker

.. autoclass:: eql.ast.Expression
.. autoclass:: eql.ast.Literal
.. autoclass:: eql.ast.TimeRange

.. autoclass:: eql.ast.Field

.. autoclass:: eql.ast.Comparison
.. autoclass:: eql.ast.InSet
.. autoclass:: eql.ast.And
.. autoclass:: eql.ast.Or
.. autoclass:: eql.ast.Not
.. autoclass:: eql.ast.FunctionCall

.. autoclass:: eql.ast.EventQuery
.. autoclass:: eql.ast.NamedSubquery
.. autoclass:: eql.ast.NamedParams
.. autoclass:: eql.ast.SubqueryBy
.. autoclass:: eql.ast.Join
.. autoclass:: eql.ast.Sequence

.. autoclass:: eql.ast.PipeCommand
.. autoclass:: eql.pipes.ByPipe
.. autoclass:: eql.pipes.HeadPipe
.. autoclass:: eql.pipes.TailPipe
.. autoclass:: eql.pipes.SortPipe
.. autoclass:: eql.pipes.UniquePipe
.. autoclass:: eql.pipes.CountPipe
.. autoclass:: eql.pipes.FilterPipe
.. autoclass:: eql.pipes.UniqueCountPipe

.. autoclass:: eql.ast.PipedQuery
.. autoclass:: eql.ast.EqlAnalytic
    :members:
