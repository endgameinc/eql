"""Test Python Engine for EQL."""
import pytest
import string

import eql
import eql.etc

try:
    from . import utils
except ImportError:
    import utils


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


def extract_tests(test_name, contents, case_sensitive, **params):
    """Extract tests from a TOML test dict."""
    for param in contents.get("params", []):
        value = type_lookup[param["type"]]()
        params[param["name"]] = eql.ast.Literal.from_python(value)

    def parametrize(expr):
        if params:
            return string.Template(expr).substitute(params)
        return expr

    for fold_test in contents.get("fold", {}).get("tests", []):
        test_info = test_name, parametrize(fold_test["expression"]), fold_test.get("expected"), case_sensitive
        fold_tests.append(test_info)

    for verifier_test in contents.get("verifier", {}).get("failures", []):
        test_info = test_name, parametrize(verifier_test["expression"])
        verifier_failures.append(test_info)

    for optimizer_test in contents.get("optimizer", {}).get("tests", []):
        test_info = test_name, parametrize(optimizer_test["expression"]), parametrize(optimizer_test["optimized"])
        optimizer_tests.append(test_info)

    for equivalent_test in contents.get("equivalent", {}).get("tests", []):
        test_info = test_name, parametrize(equivalent_test["expression"]), parametrize(equivalent_test["alternate"])
        equivalent_tests.append(test_info)

    for loop in contents.get("loop", []):
        for param in loop["param"]["type"]:
            p_loop = loop.copy()
            p_loop["params"] = [{"type": param, "name": loop["param"]["name"]}]

            extract_tests(test_name, p_loop, case_sensitive, **params)


def load_tests():
    """Load up tests from toml to prepare for pytest.mark.parametrize."""
    if fold_tests or verifier_failures:
        return

    for name in ["test_string_functions.toml", "test_folding.toml", "test_optimizer.toml"]:
        data = eql.load_dump(eql.etc.get_etc_path(name))

        for test_name, contents in sorted(data.items()):
            case_sensitive = contents.get("case_sensitive")

            extract_tests(test_name, contents, case_sensitive=case_sensitive)


load_tests()


def engine_eval(expr):
    """Evaluate an unoptimized expression in the ``PythonEngine``."""
    return eql.PythonEngine().convert(expr)(None)


@pytest.mark.parametrize("name, text, expected, case_sensitive", fold_tests)
def test_fold(name, text, expected, case_sensitive):
    """Check that expressions fold and evaluate correctly."""
    with eql.parser.skip_optimizations:
        parsed = eql.parse_expression(text)

    if case_sensitive is True:
        raise pytest.skip("Case-sensitivity not yet supported")

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
