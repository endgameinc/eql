"""Test Lucene regular expression syntax validation."""
# -*- coding: utf-8 -*-
import unittest

from eql.lucene_regex import LuceneRegexError, validate_lucene_regex


class TestLuceneRegex(unittest.TestCase):
    """Test the port of Lucene's RegExp parser."""

    def test_valid_patterns(self):
        """Test patterns that Lucene compiles."""
        patterns = [
            "",
            ".*",
            "[a-z]+",
            "[^a-z0-9_]",
            r"\d+\.\w+\s\S\W\D",
            r".*[\\/]cmd\.exe",
            r"[\d\]a-c]",
            "a{2}b{2,}c{2,5}",
            "a**",
            "(foo|bar)?baz",
            "()",
            # Lucene parses a leading operator character as a literal
            "|abc",
            "*abc",
            # escaped operators are literals
            r".*(;|\&\&|\>|\<|\~|\#|\@|\").*",
            # operators are literals inside a character class
            '[&<>~#@"]',
            # intended use of the operators
            "foo<1-100>",
            "foo<9-1>",
            "@&~(foo.*)",
            "~~abc",
            '"literal.string"',
            "#|abc",
            # interval bounds are parsed with Java's Integer.parseInt, which accepts a sign and Unicode digits
            "<+1-5>",
            u"<١-٩>",
        ]
        for pattern in patterns:
            with self.subTest(pattern=pattern):
                validate_lucene_regex(pattern)

    def test_invalid_patterns(self):
        """Test patterns that Lucene rejects, with the error Lucene raises."""
        patterns = {
            # unescaped `<` opens an interval that is never closed
            ".*(;|&&|>|<).*": "expected '>' at position 14",
            ".*<script>.*": "'script' not found",
            "a<1-b>": "interval syntax error",
            "a<-5>": "interval syntax error",
            "a<5->": "interval syntax error",
            "a<1-2-3>": "interval syntax error",
            "a< 1-2>": "interval syntax error",
            "a<1_0-20>": "interval syntax error",
            u"a<²-3>": "interval syntax error",
            "a<1-99999999999>": "interval syntax error",
            "a~": "unexpected end-of-string",
            "a&": "unexpected end-of-string",
            "a|": "unexpected end-of-string",
            '.*"quoted.*': "expected '\"'",
            "(abc": "expected '\\)'",
            "abc)": "end-of-string expected at position 3",
            "[abc": "expected '\\]'",
            "[]": "expected '\\]'",
            "[z-a]": "invalid range: from \\(122\\) cannot be > to \\(97\\)",
            "[a-]": "invalid range",
            "a{3,1}": "invalid repetition range\\(out of order\\): 3..1",
            "a{,3}": "integer expected",
            "a{3": "expected '}'",
            "a{99999999999}": "For input string",
            r"\q": "invalid character class \\\\q",
            r"[\q]": "invalid character class \\\\q",
            "abc\\": "unexpected end-of-string",
        }
        for pattern, message in patterns.items():
            with self.subTest(pattern=pattern):
                with self.assertRaisesRegex(LuceneRegexError, message) as ctx:
                    validate_lucene_regex(pattern)
                self.assertEqual(ctx.exception.pattern, pattern)
