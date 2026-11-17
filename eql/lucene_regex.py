"""Syntax validation for Lucene regular expressions.

Elasticsearch compiles EQL ``regex``/``regex~`` patterns with Lucene's ``RegExp`` and every optional operator enabled
(``RegExp.ALL | RegExp.DEPRECATED_COMPLEMENT``). That makes ``&``, ``<``, ``>``, ``~``, ``#``, ``@`` and ``"``
operators rather than literals, so a string that EQL accepts as a pattern can still be rejected by Elasticsearch, or
silently match something else.

This module is a validation-only port of the parser in ``org.apache.lucene.util.automaton.RegExp``, which accepts the
same syntax in Lucene 9.x (Elasticsearch 8.x) and Lucene 10.x (Elasticsearch 9.x). It raises the errors Lucene raises
while parsing. It does not build automata, so failures that only happen during determinization (for example, a
pattern that is too complex) are not detected.

See: https://www.elastic.co/docs/reference/query-languages/query-dsl/regexp-syntax
"""

__all__ = (
    "LuceneRegexError",
    "validate_lucene_regex",
)

# After ``\``, Lucene expands these characters as a predefined class (``\d``, ``\s``, ``\w`` and their negations),
# an escaped backslash, or an invalid escape.
_ESCAPE_CLASS_CHARS = "\\ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz"
# Letters that do not name a predefined class are an error after ``\``, as in ``java.util.regex.Pattern``.
_INVALID_ESCAPE_CHARS = "abcefghijklmnopqrtuvxyzABCEFGHIJKLMNOPQRTUVXYZ"
_DIGITS = "0123456789"
_JAVA_INT_MIN = -2 ** 31
_JAVA_INT_MAX = 2 ** 31 - 1


class LuceneRegexError(ValueError):
    """A pattern that Lucene's regular expression parser rejects."""

    def __init__(self, message, pattern):
        """Create the error with Lucene's message for the rejected pattern."""
        self.pattern = pattern
        super(LuceneRegexError, self).__init__(message)


def _java_parse_int(value):
    """Return ``Integer.parseInt(value)``, or None where Java throws ``NumberFormatException``.

    Java accepts a leading sign and any Unicode decimal digit, but not whitespace or underscores like ``int()`` does.
    """
    digits = value[1:] if value.startswith(("+", "-")) else value
    if not digits or not digits.isdecimal():
        return None

    number = int(value)
    if not _JAVA_INT_MIN <= number <= _JAVA_INT_MAX:
        return None
    return number


class _LuceneRegexParser(object):
    """Recursive descent parser mirroring ``RegExp`` with all syntax flags and complement enabled.

    Method names follow the Lucene parser so the two can be compared side by side.
    """

    def __init__(self, pattern):
        self.pattern = pattern
        self.pos = 0

    def error(self, message):
        return LuceneRegexError(message, self.pattern)

    def more(self):
        return self.pos < len(self.pattern)

    def peek(self, chars):
        return self.more() and self.pattern[self.pos] in chars

    def match(self, char):
        if self.more() and self.pattern[self.pos] == char:
            self.pos += 1
            return True
        return False

    def next(self):
        if not self.more():
            raise self.error("unexpected end-of-string")
        char = self.pattern[self.pos]
        self.pos += 1
        return char

    def parse(self):
        # The RegExp constructor maps the empty pattern to the empty string before parsing.
        if not self.pattern:
            return
        self.parse_union_exp()
        if self.more():
            raise self.error("end-of-string expected at position {}".format(self.pos))

    def parse_union_exp(self):
        self.parse_inter_exp()
        while self.match("|"):
            self.parse_inter_exp()

    def parse_inter_exp(self):
        self.parse_concat_exp()
        while self.match("&"):
            self.parse_concat_exp()

    def parse_concat_exp(self):
        self.parse_repeat_exp()
        while self.more() and not self.peek(")|&"):
            self.parse_repeat_exp()

    def parse_repeat_exp(self):
        self.parse_compl_exp()
        while self.peek("?*+{"):
            if not self.match("{"):
                self.next()
                continue

            start = self.pos
            while self.peek(_DIGITS):
                self.next()
            if start == self.pos:
                raise self.error("integer expected at position {}".format(self.pos))
            n = self.parse_repeat_bound(start)
            m = n
            if self.match(","):
                start = self.pos
                while self.peek(_DIGITS):
                    self.next()
                m = self.parse_repeat_bound(start) if start != self.pos else None
            if not self.match("}"):
                raise self.error("expected '}}' at position {}".format(self.pos))
            if m is not None and n > m:
                raise self.error("invalid repetition range(out of order): {}..{}".format(n, m))

    def parse_repeat_bound(self, start):
        value = self.pattern[start:self.pos]
        number = _java_parse_int(value)
        if number is None:
            raise self.error('For input string: "{}"'.format(value))
        return number

    def parse_compl_exp(self):
        # Lucene 10 deprecates ``~`` (complement), but Elasticsearch still enables it with DEPRECATED_COMPLEMENT.
        while self.match("~"):
            pass
        self.parse_char_class_exp()

    def parse_char_class_exp(self):
        if self.match("["):
            self.match("^")
            self.parse_char_classes()
            if not self.match("]"):
                raise self.error("expected ']' at position {}".format(self.pos))
        else:
            self.parse_simple_exp()

    def parse_char_classes(self):
        while True:
            if self.match("\\"):
                if self.peek(_ESCAPE_CLASS_CHARS):
                    self.expand_predefined()
                else:
                    self.next()
            else:
                start = self.parse_char_exp()
                if self.match("-"):
                    end = self.parse_char_exp()
                    if start > end:
                        raise self.error(
                            "invalid range: from ({}) cannot be > to ({})".format(ord(start), ord(end)))

            if not self.more() or self.peek("]"):
                break

    def expand_predefined(self):
        if self.peek(_INVALID_ESCAPE_CHARS):
            raise self.error("invalid character class \\" + self.next())
        self.next()

    def parse_simple_exp(self):
        if self.match(".") or self.match("#") or self.match("@"):
            return

        if self.match('"'):
            while self.more() and not self.peek('"'):
                self.next()
            if not self.match('"'):
                raise self.error("expected '\"' at position {}".format(self.pos))
        elif self.match("("):
            if self.match(")"):
                return
            self.parse_union_exp()
            if not self.match(")"):
                raise self.error("expected ')' at position {}".format(self.pos))
        elif self.match("<"):
            self.parse_interval()
        elif self.match("\\") and self.peek(_ESCAPE_CLASS_CHARS):
            self.expand_predefined()
        else:
            # Either a plain character, or the character after a ``\`` that was consumed above.
            self.next()

    def parse_interval(self):
        start = self.pos
        while self.more() and not self.peek(">"):
            self.next()
        if not self.match(">"):
            raise self.error("expected '>' at position {}".format(self.pos))

        body = self.pattern[start:self.pos - 1]
        dash = body.find("-")
        if dash == -1:
            # ``<name>`` refers to a named automaton. Elasticsearch never provides one, so Lucene cannot resolve it.
            raise self.error("'{}' not found (named automata like <{}> are not supported)".format(body, body))

        # Exactly one `-`, with a Java int on each side (an empty side is not an int)
        smin, smax = body[:dash], body[dash + 1:]
        if "-" in smax or _java_parse_int(smin) is None or _java_parse_int(smax) is None:
            raise self.error("interval syntax error at position {}".format(self.pos - 1))

    def parse_char_exp(self):
        self.match("\\")
        return self.next()


def validate_lucene_regex(pattern):
    """Raise :class:`LuceneRegexError` if Lucene's regular expression parser would reject ``pattern``.

    :param str pattern: The regular expression, after EQL string unescaping
    :raises LuceneRegexError: If Elasticsearch would fail to compile the pattern
    """
    _LuceneRegexParser(pattern).parse()
