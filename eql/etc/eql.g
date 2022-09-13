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
sequence: "sequence" [join_values with_params? | with_params join_values?] subquery_by+ [until_subquery_by]
join: "join" join_values? subquery_by subquery_by+ until_subquery_by?
until_subquery_by.2: "until" subquery_by
pipes: pipe+
pipe: "|" name [single_atom single_atom+ | expressions]

join_values.2: "by" expressions
?with_params.2: "with" "maxspan" EQUALS time_range
repeated_sequence.2: "with" "runs" EQUALS UNSIGNED_INTEGER
sequence_alias.2: "as" name
time_range: number name?


subquery_by: subquery fork_param? join_values? repeated_sequence? sequence_alias?
subquery: "[" event_query "]"
fork_param: "fork" (EQUALS boolean)?

// Expressions
expressions: expr ("," expr)* [","]
?expr: or_expr
?or_expr: and_expr ("or" and_expr)*
?and_expr: not_expr ("and" not_expr)*
?not_expr.3: NOT_OP* term
?term: sum_expr comp_op sum_expr -> comparison
     | sum_expr "not" IN "(" expressions [","]? ")"  -> not_in_set
     | sum_expr IN "(" expressions [","]? ")" -> in_set
     | sum_expr STRING_PREDICATE (literal | "(" literal ("," literal)* ")") -> string_predicate
     | sum_expr


// Need to recover these tokens
IN.3: "in~" | "in"
EQUALS: "==" | "="
STRING_PREDICATE.3:  ":"
                  |  "like~"
                  |  "regex~"
                  |  "like"
                  |  "regex"
COMP_OP: "<=" | "<" | "!=" | ">=" | ">"
?comp_op: EQUALS | COMP_OP
MULT_OP:    "*" | "/" | "%"
NOT_OP:     "not"

?sum_expr: mul_expr (SIGN mul_expr)*
?mul_expr: named_subquery_test (MULT_OP named_subquery_test)*
?named_subquery_test: named_subquery
                    | method_chain
named_subquery.2: name "of" subquery
?method_chain: value method*
?value: SIGN? function_call
      | SIGN? atom

// hacky approach to work around this ambiguity introduced with the colon operator
// x : length
// x : length( ) not allowed, now requires `:length(` form
METHOD_START.3: ":" NAME "("
method_name: METHOD_START
method: method_name [expressions] ")"
function_call: (INSENSITIVE_NAME | NAME) "(" [expressions] ")"
?atom: single_atom
     |  "(" expr ")"
?signed_single_atom: SIGN? single_atom
?single_atom: literal
            | varpath
            | field
            | base_field
base_field: name | escaped_name
field: FIELD
      | OPTIONAL_FIELD
literal: number
       | boolean
       | null
       | string
!boolean: "true"
        | "false"
null: "null"
number: UNSIGNED_INTEGER
      | DECIMAL
string: RAW_TQ_STRING
      | DQ_STRING
      | SQ_STRING
      | RAW_DQ_STRING
      | RAW_SQ_STRING
varpath: "$" (field | base_field)

// Check against keyword usage
name: NAME
escaped_name: ESCAPED_NAME

// Tokens
// pin the first "." or "[" to resolve token ambiguities
// sequence by pid [1] [true] looks identical to:
// sequence by pid[1] [true]
FIELD: FIELD_IDENT (ATTR | INDEX)+
OPTIONAL_FIELD: "?" FIELD_IDENT (ATTR | INDEX)*
ATTR: "." WHITESPACE? FIELD_IDENT
INDEX: "[" WHITESPACE? UNSIGNED_INTEGER WHITESPACE? "]"
FIELD_IDENT: NAME | ESCAPED_NAME

// create a non-conflicting helper rule to deconstruct
field_parts: field_ident ("." field_ident | "[" array_index "]")*
!array_index: UNSIGNED_INTEGER
!field_ident: NAME | ESCAPED_NAME


LCASE_LETTER: "a".."z"
UCASE_LETTER: "A".."Z"
DIGIT: "0".."9"

LETTER: UCASE_LETTER | LCASE_LETTER
WORD: LETTER+

ESCAPED_NAME: "`" /[^`\r\n]+/ "`"
INSENSITIVE_NAME.2: ("_"|LETTER) ("_"|LETTER|DIGIT)* "~"
NAME: ("_"|LETTER) ("_"|LETTER|DIGIT)*
UNSIGNED_INTEGER: /[0-9]+/
EXPONENT: /[Ee][-+]?\d+/
DECIMAL: UNSIGNED_INTEGER? "." UNSIGNED_INTEGER+ EXPONENT?
       | UNSIGNED_INTEGER EXPONENT
SIGN:           "+" | "-"
DQ_STRING:        /"(\\[btnfr"'\\]|\\u\{[a-zA-Z0-9]{2,8}\}|[^\r\n"\\])*"/
SQ_STRING:        /'(\\[btnfr"'\\]|[^\r\n'\\])*'/
RAW_DQ_STRING:    /\?"(\\\"|[^"\r\n])*"/
RAW_SQ_STRING:    /\?'(\\\'|[^'\r\n])*'/
RAW_TQ_STRING.2:  /"""[^\r\n]*?""""?"?/

%import common.NEWLINE

COMMENT: "//" /[^\n]*/
ML_COMMENT: "/*" /(.|\n|\r)*?/ "*/"
WHITESPACE: (" " | "\r" | "\n" | "\t" )+

%ignore COMMENT
%ignore ML_COMMENT
%ignore WHITESPACE
