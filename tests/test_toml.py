"""Test Python Engine for EQL."""
import contextlib
import pytest
import string

import eql
import eql.etc

from . import utils


fold_tests = []
verifier_failures = []
optimizer_tests = []
equivalent_tests = []

type_lookup = {
    "null": lambda: None,
    "bool": utils.random_bool,
    "int": utils.random_int,
    "string": utils.random_string,
    "float": utils.random_float,
}


def extract_tests(test_name, contents, case_settings, **params):
    """Extract tests from a TOML test dict."""
    for param in contents.get("params", []):
        value = type_lookup[param["type"]]()
        params[param["name"]] = eql.ast.Literal.from_python(value)

    def parametrize(expr):
        if params:
            return string.Template(expr).substitute(params)
        return expr

    for fold_test in contents.get("fold", {}).get("tests", []):
        expression = parametrize(fold_test["expression"])
        expected = fold_test.get("expected", None)

        for case_sensitive in case_settings:
            fold_tests.append((test_name, expression, expected, case_sensitive))

    for verifier_test in contents.get("verifier", {}).get("failures", []):
        test_info = test_name, parametrize(verifier_test["expression"])
        verifier_failures.append(test_info)

    for optimizer_test in contents.get("optimizer", {}).get("tests", []):
        expression = parametrize(optimizer_test["expression"])
        expected = parametrize(optimizer_test["optimized"])

        optimizer_tests.append((test_name, expression, expected))

    for equivalent_test in contents.get("equivalent", {}).get("tests", []):
        test_info = test_name, parametrize(equivalent_test["expression"]), parametrize(equivalent_test["alternate"])
        equivalent_tests.append(test_info)

    for loop in contents.get("loop", []):
        for param in loop["param"]["type"]:
            p_loop = loop.copy()
            p_loop["params"] = [{"type": param, "name": loop["param"]["name"]}]

            extract_tests(test_name, p_loop, case_settings, **params)


def load_tests():
    """Load up tests from toml to prepare for pytest.mark.parametrize."""
    if fold_tests or verifier_failures:
        return

    for name in ["test_string_functions.toml", "test_folding.toml", "test_optimizer.toml"]:
        data = eql.load_dump(eql.etc.get_etc_path(name))

        for test_name, contents in sorted(data.items()):
            test_name = "{file}:{test}".format(file=name, test=test_name)
            case_settings = []

            if "case_sensitive" not in contents and "case_insensitive" not in contents:
                case_sensitive = True
                case_insensitive = True
            else:
                case_sensitive = contents.get("case_sensitive") is True
                case_insensitive = contents.get("case_insensitive") is True

            if case_sensitive:
                case_settings.append(True)

            if case_insensitive:
                case_settings.append(False)

            assert len(case_settings) > 0, "{test} is missing case_sensitive/case_insensitive".format(test=test_name)

            extract_tests(test_name, contents, case_settings)


load_tests()


def engine_eval(expr):
    """Evaluate an unoptimized expression in the ``PythonEngine``."""
    return eql.PythonEngine().convert(expr)(None)


@contextlib.contextmanager
def case_sensitivity(enabled):
    """Helper function for toggling case sensitivity."""
    prev = eql.utils.CASE_INSENSITIVE

    try:
        eql.utils.CASE_INSENSITIVE = not enabled
        yield
    finally:
        eql.utils.CASE_INSENSITIVE = prev


@pytest.mark.parametrize("name, text, expected, case_sensitive", fold_tests)
def test_fold(name, text, expected, case_sensitive):
    """Check that expressions fold and evaluate correctly."""
    with case_sensitivity(case_sensitive):
        with eql.parser.skip_optimizations:
            parsed = eql.parse_expression(text)

        assert not isinstance(parsed, eql.ast.Literal)
        assert engine_eval(parsed) == expected
        assert parsed.fold() == expected


@pytest.mark.parametrize("name, text", verifier_failures)
def test_verifier(name, text):
    """Check that invalid function signatures are correctly detected."""
    with pytest.raises(eql.EqlSemanticError):
        eql.parse_expression(text)


@pytest.mark.parametrize("name, expression, alternate", equivalent_tests)
def test_equivalent(name, expression, alternate):
    """Check that two expressions parse to the same AST."""
    with eql.parser.skip_optimizations:
        source_ast = eql.parse_expression(expression)
        dest_ast = eql.parse_expression(alternate)

        assert source_ast == dest_ast


@pytest.mark.parametrize("name, unoptimized, optimized", optimizer_tests)
def test_optimizer(name, unoptimized, optimized):
    """Check that optimization rules are working as expected."""
    with eql.parser.skip_optimizations:
        unoptimized_ast = eql.parse_expression(unoptimized)
        optimized_ast = eql.parse_expression(optimized)

        assert unoptimized_ast != optimized_ast
        assert unoptimized_ast.optimize(recursive=True) == optimized_ast
