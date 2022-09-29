# NRPyLaTeX EBNF Grammar

```
<LATEX>         -> ( '%' <MACRO> | [ '%' ] <ASSIGNMENT> ) { [ <RETURN> ] ( '%' <MACRO> | [ '%' ] <ASSIGNMENT> ) }*
<MACRO>         -> <DEFINE> | <ASSIGN> | <IGNORE> | <SREPL> | <COORD> | <INDEX>
<DEFINE>        -> <DEFINE_MACRO> { <VARIABLE> }+ { '--' ( <ZERO> | <KRON> | <CONST> | <OPTION> ) }*
<ASSIGN>        -> <ASSIGN_MACRO> { <VARIABLE> }+ { '--' <OPTION> }+
<IGNORE>        -> <IGNORE_MACRO> { <STRING> }+
<SREPL>         -> <SREPL_MACRO> <STRING> <ARROW> <STRING> [ '--' <PERSIST> ]
<COORD>         -> <COORD_MACRO> ( <LBRACK> <SYMBOL> [ ',' <SYMBOL> ]* <RBRACK> | '--' <DEFAULT> )
<INDEX>         -> <INDEX_MACRO> [ { <LETTER> | '[' <LETTER> '-' <LETTER> ']' }+ | '--' <DEFAULT> ] '--' <DIM> <INTEGER>
<OPTION>        -> <DIM> <INTEGER> | <SYM> <SYMMETRY> | <WEIGHT> <NUMBER> | <DERIV> <SUFFIX> | <METRIC> [ <VARIABLE> ]
<ASSIGNMENT>    -> <OPERATOR> = <EXPRESSION> [ '\\' ] [ '%' <NOIMPSUM> ]
<EXPRESSION>    -> <TERM> { ( '+' | '-' ) <TERM> }*
<TERM>          -> <FACTOR> { [ '/' ] <FACTOR> }*
<FACTOR>        -> <BASE> { '^' <EXPONENT> }*
<BASE>          -> [ '-' ] ( <NUMBER> | <COMMAND> | <OPERATOR> | <SUBEXPR> )
<EXPONENT>      -> <BASE> | '{' <EXPRESSION> '}' | '{' '{' <EXPRESSION> '}' '}'
<SUBEXPR>       -> '(' <EXPRESSION> ')' | '[' <EXPRESSION> ']' | '\' '{' <EXPRESSION> '\' '}'
<COMMAND>       -> <FUNC> | <FRAC> | <SQRT> | <NLOG> | <TRIG>
<FUNC>          -> <FUNC_CMD> <SUBEXPR>
<FRAC>          -> <FRAC_CMD> '{' <EXPRESSION> '}' '{' <EXPRESSION> '}'
<SQRT>          -> <SQRT_CMD> [ '[' <INTEGER> ']' ] '{' <EXPRESSION> '}'
<NLOG>          -> <NLOG_CMD> [ '_' ( <NUMBER> | '{' <NUMBER> '}' ) ] ( <NUMBER> | <TENSOR> | <SUBEXPR> )
<TRIG>          -> <TRIG_CMD> [ '^' ( <NUMBER> | '{' <NUMBER> '}' ) ] ( <NUMBER> | <TENSOR> | <SUBEXPR> )
<OPERATOR>      -> [ '%' <DERIV> <SUFFIX> ] ( <PARDRV> | <COVDRV> | <LIEDRV> | <TENSOR> )
<PARDRV>        -> <PAR_SYM> '_' <INDEXING_2> ( <OPERATOR> | <SUBEXPR> )
<COVDRV>        -> ( <COV_SYM> | <DIACRITIC> '{' <COV_SYM> '}' ) ( '^' | '_' ) <INDEXING_2> ( <OPERATOR> | <SUBEXPR> )
<LIEDRV>        -> <LIE_SYM> '_' <SYMBOL> ( <OPERATOR> | <SUBEXPR> )
<TENSOR>        -> <SYMBOL> [ ( '_' <INDEXING_4> ) | ( '^' <INDEXING_3> [ '_' <INDEXING_4> ] ) ]
<SYMBOL>        -> <LETTER> | <DIACRITIC> '{' <SYMBOL> '}' | <TEXT_CMD> '{' <LETTER> { '_' | <LETTER> | <INTEGER> }* '}'
<INDEXING_1>    -> <LETTER> [ '_' <INDEXING_2> ] | <INTEGER>
<INDEXING_2>    -> <LETTER> | <INTEGER> | '{' <INDEXING_1> '}'
<INDEXING_3>    -> <INDEXING_2> | '{' { <INDEXING_1> }+ '}'
<INDEXING_4>    -> <INDEXING_2> | '{' ( ',' | ';' ) { <INDEXING_1> }+ | { <INDEXING_1> }+ [ ( ',' | ';' ) { <INDEXING_1> }+ ] '}'
<VARIABLE>      -> <LETTER> { <LETTER> | <UNDERSCORE> }*
<NUMBER>        -> <RATIONAL> | <DECIMAL> | <INTEGER> | <PI>
```
