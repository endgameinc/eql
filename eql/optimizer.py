"""Optimizer for the EQL syntax tree."""
from .functions import get_function
from .utils import is_string
from .walkers import Walker, DepthFirstWalker

__all__ = (
    "Optimizer",
)


class Optimizer(DepthFirstWalker):
    """Class for optimizing AST nodes."""

    def __init__(self, recursive=False):
        """Class for optimizing AST nodes."""
        Walker.__init__(self)
        self.recursive = recursive

    def walk(self, node, *args, **kwargs):
        """Override the default walk to optionally recurse."""
        if self.recursive:
            return DepthFirstWalker.walk(self, node, *args, **kwargs)

        return Walker.walk(self, node, *args, **kwargs)

    @staticmethod
    def _walk_and(node):
        terms = []
        current = node.terms[0]
        for term in node.terms[1:]:
            current = current & term
            if isinstance(current, And):
                terms.extend(current.terms[:-1])
                current = current.terms[-1]

        if terms:
            terms.append(current)
            return And(terms)
        return current

    @staticmethod
    def _walk_comparison(node):
        if isinstance(node.left, Literal) and isinstance(node.right, Literal):
            lhs = node.left.value
            rhs = node.right.value

            # Check that the types match first
            if not isinstance(node.right, type(node.left)):
                return Boolean(node.comparator == Comparison.NE)

            if isinstance(node.left, String):
                lhs = lhs.lower()
                rhs = rhs.lower()

            return Boolean(node.function(lhs, rhs))

        # assumes calling the same function twice with the same args returns the same result
        elif node.left == node.right:
            return Boolean(node.comparator in (Comparison.EQ, Comparison.LE, Comparison.GE))

        return node

    @staticmethod
    def _walk_function_call(node):  # type: (FunctionCall) -> EqlNode
        func = get_function(node.name)
        arguments = [arg.optimize() for arg in node.arguments]

        if func and all(isinstance(arg, Literal) for arg in arguments):
            try:
                rv = func.run(*[arg.value for arg in arguments])
                return Literal.from_python(rv)
            except NotImplementedError:
                pass

        return FunctionCall(node.name, arguments, node.as_method)

    @staticmethod
    def _walk_in_set(node):
        expression = node.expression

        # move all the literals to the front, preserve their ordering
        literals = [v for k, v in node.get_literals().items()]
        dynamic = [v for v in node.container if not isinstance(v, Literal)]
        container = literals + dynamic

        # check to see if a literal value is in the list of literal values
        if isinstance(node.expression, Literal):
            value = node.expression.value
            if is_string(value):
                value = value.lower()
            if value in node.get_literals():
                return Boolean(True)
            container = dynamic

        if len(container) == 0:
            return Boolean(False)
        elif len(container) == 1:
            return Comparison(expression, Comparison.EQ, container[0]).optimize()
        elif expression in container:
            return Boolean(True)

        return InSet(expression, container)

    @staticmethod
    def _walk_math_operation(node):
        left = node.left.optimize()
        right = node.right.optimize()

        if isinstance(left, Number) and isinstance(right, Number):
            # don't divide by zero when optimizing, leave that to the target implementation
            if not (right.value == 0 and node.operator in ("/", "%")):
                return Number(node.func(left.value, right.value))

        if isinstance(right, MathOperation) and right.left == Number(0):
            # a +- b parses as a + (0 - b) should become a + -b
            if node.operator in ("-", "+") and right.operator in ("-", "+"):
                operator = "-" if (node.operator == "-") ^ (right.operator == "-") else "+"
                return MathOperation(left, operator, right.right)

        return MathOperation(left, node.operator, right)

    @staticmethod
    def _walk_not(node):
        optimized_term = node.term.optimize()
        return ~ optimized_term

    @staticmethod
    def _walk_or(node):
        terms = []
        current = node.terms[0]
        for term in node.terms[1:]:
            current = current | term
            if isinstance(current, Or):
                terms.extend(current.terms[:-1])
                current = current.terms[-1]

        if terms:
            terms.append(current)
            return Or(terms)
        return current


from .ast import *  # noqa: E402
