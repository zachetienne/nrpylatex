# NRPyLaTeX

[NRPy+](https://github.com/zachetienne/nrpytutorial)'s LaTeX Interface to SymPy (CAS) for Numerical Relativity

- automatic expansion of [Einstein notation](https://en.wikipedia.org/wiki/Einstein_notation)
- automatic generation of Levi-Civita symbols
- automatic generation of Christoffel symbols
- automatic generation of covariant derivatives
- automatic generation of Lie derivatives
- automatic generation of metric inverse and determinant
- automatic index raising/lowering using the metric
- robust exception handling (including invalid indexing)

## Alternate CAS (Computer Algebra System)

If you are using Mathematica instead of SymPy,

        from sympy import mathematica_code
        
        namespace = parse(...)
        for var in namespace:
            exec(f'{var} = mathematica_code({var})') # Python 3

If you are using a different CAS, reference the SymPy [documentation]((https://docs.sympy.org/latest/modules/printing.html)) to find the relevant printing function.

## Installation

To install NRPyLaTeX from [PyPI](https://test.pypi.org/project/nrpylatex/), run the following command

    $ pip install -i https://test.pypi.org/simple/ nrpylatex

## Interactive Tutorial (MyBinder)

[Getting Started](https://mybinder.org/v2/gh/zachetienne/nrpytutorial/HEAD?filepath=Tutorial-LaTeX_Parser_Interface.ipynb) | [Guided Example (Cartesian BSSN)](https://mybinder.org/v2/gh/zachetienne/nrpytutorial/HEAD?filepath=Tutorial-LaTeX_Interface_Example-BSSN_Cartesian.ipynb)

## Documentation and Usage

### PARSE MACRO
    parse - parse an equation without rendering

    USAGE
        parse EQUATION

    EQUATION
        syntax: (tensorial) LaTeX equation

### SREPL MACRO
    srepl - syntactic string replacement

    USAGE
        srepl RULE, ...
    
    RULE
        syntax: "..." -> "..."
        remark (1): string A and string B are considered equal
            if they are equivalent syntactically (i.e. lexeme, token)
        remark (2): substring can include a capture group
            single token capture group <#>
            continuous capture group <#..>

### VARDEF MACRO
    vardef - define a variable

    USAGE
        vardef [OPERAND ...] [OPTION] VARIABLE, ... [DIMENSION]

    OPERAND
        metric=VARIABLE
            desc: assign metric to variable for automatic index raising/lowering
            default: metric associated with diacritic (or lack thereof)

        weight=NUMBER
            desc: assign weight to variable (used to generate Lie derivative)
            default: 0

        diff_type=DIFF_TYPE
            desc: assign derivative type to variable {symbolic | dD (numeric) | dupD (upwind)}
            default: symbolic

        symmetry=SYMMETRY
            desc: assign (anti)symmetry to variable
            default: nosym
            example(s):  sym01 -> [i][j] = [j][i], anti01 -> [i][j] = -[j][i]

    OPTION
        const
            label variable type: constant

        kron
            label variable type: delta function

        metric
            label variable type: metric

        zero
            zero each component of variable

    VARIABLE
        syntax: alphabetic string inside of '...'
        example(s): 'vU', 'gDD', 'alpha'

    DIMENSION
        syntax: variable dimension inside of (...)
        default: 3D

### KEYDEF MACRO
    vardef - define a keyword

    USAGE
        keydef OPERAND VARIABLE
        <BASIS_KWRD> <BASIS> | <INDEX_KWRD> <INDEX>

    OPERAND
        basis=BASIS
            desc: define basis (or coordinate system)
            example(s): [x, y, z], [r, \phi]

        index=RANGE
            desc: override index range
            example(s): i (4D), [a-z] (2D)

    VARIABLE
        syntax: alphabetic string inside of '...'
        example(s): 'vU', 'gDD', 'alpha'

### ASSIGN MACRO
    assign - assign property(ies) to a variable

    USAGE
        assign [OPERAND ...] VARIABLE, ...

    OPERAND
        metric=VARIABLE
            desc: assign metric to variable for automatic index raising/lowering
            default: metric associated with diacritic (or lack thereof)

        weight=NUMBER
            desc: assign weight to variable (used to generate Lie derivative)
            default: 0

        diff_type=DIFF_TYPE
            desc: assign derivative type to variable {symbolic | dD (numeric) | dupD (upwind)}
            default: symbolic

        symmetry=SYMMETRY
            desc: assign (anti)symmetry to variable
            default: nosym
            example(s):  sym01 -> [i][j] = [j][i], anti01 -> [i][j] = -[j][i]

    VARIABLE
        alphabetic string inside of '...'
        example(s): 'vU', 'gDD', 'alpha'

### IGNORE MACRO
    ignore - remove a substring; equivalent to srepl "..." -> "" (empty replacement)

    USAGE
        ignore SUBSTRING, ...
    
    SUBSTRING
        syntax: "..."
