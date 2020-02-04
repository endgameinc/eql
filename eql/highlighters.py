"""Highlighters for EQL."""
from pygments.lexer import RegexLexer, bygroups, include
from pygments import token
from eql.functions import list_functions
from eql.pipes import list_pipes


class EqlLexer(RegexLexer):
    """Pygments Lexer for EQL."""

    name = 'Event Query Language'
    aliases = ['eql']
    filenames = ['*.eql']

    _sign = r'[\-+]'
    _integer = r'\d+'
    _float = r'\d*\.\d+([Ee][-+]?\d+)?'
    _time_units = r's|sec\w+|m|min\w+|h|hour|hr|d|day'
    _name = r'[a-zA-Z][_a-zA-Z0-9]*'
    _pipe_names = set(list_pipes())

    tokens = {
        'comments': [
            (r'//(\n|[\w\W]*?[^\\]\n)', token.Comment.Single),
            (r'/[*][\w\W]*?[*]/', token.Comment.Multiline),
            (r'/[*][\w\W]*', token.Comment.Multiline),
        ],
        'whitespace': [
            (r'\s+', token.Whitespace),
        ],
        'root': [
            include('whitespace'),
            include('comments'),
            (r'(and|in|not|or)\b', token.Operator.Word),  # Keyword.Pseudo can also work
            (r'(join|sequence|until|where)\b', token.Keyword),
            (r'(%s)(=\s+)(where)\b' % _name, bygroups(token.Name, token.Whitespace, token.Keyword)),
            (r'(const)(\s+)(%s)\b' % _name, bygroups(token.Keyword.Declaration, token.Whitespace, token.Name.Constant)),
            (r'(macro)(\s+)(%s)\b' % _name, bygroups(token.Keyword.Declaration, token.Whitespace, token.Name.Constant)),
            (r'(by|of|with)\b', token.Keyword.QueryModifier),
            (r'(true|false|null)\b', token.Name.Builtin),

            # built in pipes
            (r'(\|)(\s*)(%s)' % '|'.join(reversed(sorted(_pipe_names, key=len))),
             bygroups(token.Operator, token.Whitespace, token.Name.Function.Magic)),

            # built in functions
            (r'(%s)(\s*\()' % '|'.join(list_functions()), bygroups(token.Name.Function, token.Text)),

            # all caps names
            (r'[A-Z][_A-Z0-9]+\b', token.Name.Other),
            (_name, token.Name),

            # time units
            (r'(%s|%s)[ \t]*(%s)\b' % (_float, _integer, _time_units), token.Literal.Date),

            (_sign + '?' + _float, token.Number.Float),
            (_sign + '?' + _integer, token.Number.Integer),

            # Continue matching strings until they are closed
            (r'"(\\[btnfr"\'\\]|[^\r\n"\\])*"?', token.String),
            (r"'(\\[btnfr'\"\\]|[^\r\n'\\])*'?", token.String),
            (r'\?"(\\"|[^"])*"?', token.String.Regex),
            (r"\?'(\\'|[^'])*'?", token.String.Regex),

            (r'(==|=|!=|<|<=|>=|>|\+|\-|\*|/|\%|:)', token.Operator),
            (r'[()\[\],.]', token.Punctuation),
        ]
    }
