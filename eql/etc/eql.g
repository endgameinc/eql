definitions: definition*
?definition: macro | constant

macro:    "macro" name "(" [name ("," name)*] ")" expr
constant: "const" name EQUALS literal

query_with_definitions: definitions piped_query
piped_query: base_query [pipes]
           | pipes
base_query: sequence
          | join
          | event_query
event_query: [name "where"] expr
sequence: "sequence" [join_values with_params? | with_params join_values?] subquery_by subquery_by+ [until_subquery_by]
join: "join" join_values? subquery_by subquery_by+ until_subquery_by?
until_subquery_by.2: "until" subquery_by
pipes: pipe+
pipe: "|" name [single_atom single_atom+ | expressions]

join_values.2: "by" expressions
?with_params.2: "with" named_params
kv: name [EQUALS (time_range | atom)]
time_range: number name
named_params: kv ("," kv)*
subquery_by: subquery named_params? join_values?
subquery: "[" event_query "]"


// Expressions
expressions: expr ("," expr)* [","]
?expr: or_expr
?or_expr: and_expr ("or" and_expr)*
?and_expr: not_expr ("and" not_expr)*
?not_expr.3: NOT_OP* term
?term: sum_expr comp_op sum_expr -> comparison
     | sum_expr "not" "in" "(" expressions [","]? ")"  -> not_in_set
     | sum_expr "in" "(" expressions [","]? ")" -> in_set
     | sum_expr

// Need to recover these tokens
EQUALS: "==" | "="
COMP_OP: "<=" | "<" | "!=" | ">=" | ">"
?comp_op: EQUALS | COMP_OP
MULT_OP:    "*" | "/" | "%"
NOT_OP:     "not"

method: ":" name "(" [expressions] ")"


?sum_expr: mul_expr (SIGN mul_expr)*
?mul_expr: named_subquery_test (MULT_OP named_subquery_test)*
?named_subquery_test: named_subquery
                    | method_chain
named_subquery.2: name "of" subquery
?method_chain: value (":" function_call)*
?value: SIGN? function_call
      | SIGN? atom
function_call.2: name "(" [expressions] ")"
?atom: single_atom
     |  "(" expr ")"
?signed_single_atom: SIGN? single_atom
?single_atom: literal
            | field
            | base_field
base_field: name
field: FIELD
literal: number
       | string
number: UNSIGNED_INTEGER
      | DECIMAL
string: DQ_STRING
      | SQ_STRING
      | RAW_DQ_STRING
      | RAW_SQ_STRING

// Check against keyword usage
name: NAME

// Tokens
// make this a token to avoid ambiguity, and make more rigid on whitespace
// sequence by pid [1] [true] looks identical to:
// sequence by pid[1] [true]
FIELD: NAME ("." WHITESPACE* NAME | "[" WHITESPACE* UNSIGNED_INTEGER WHITESPACE* "]")+
LCASE_LETTER: "a".."z"
UCASE_LETTER: "A".."Z"
DIGIT: "0".."9"

LETTER: UCASE_LETTER | LCASE_LETTER
WORD: LETTER+

NAME: ("_"|LETTER) ("_"|LETTER|DIGIT)*
UNSIGNED_INTEGER: /[0-9]+/
EXPONENT: /[Ee][-+]?\d+/
DECIMAL: UNSIGNED_INTEGER? "." UNSIGNED_INTEGER+ EXPONENT?
       | UNSIGNED_INTEGER EXPONENT
SIGN:           "+" | "-"
DQ_STRING:      /"(\\[btnfr"'\\]|[^\r\n"\\])*"/
SQ_STRING:      /'(\\[btnfr"'\\]|[^\r\n'\\])*'/
RAW_DQ_STRING:  /\?"(\\\"|[^"\r\n])*"/
RAW_SQ_STRING:  /\?'(\\\'|[^'\r\n])*'/

%import common.NEWLINE

COMMENT: "//" /[^\n]*/
ML_COMMENT: "/*" /(.|\n|\r)*?/ "*/"
WHITESPACE: (" " | "\r" | "\n" | "\t" )+

%ignore COMMENT
%ignore ML_COMMENT
%ignore WHITESPACE
