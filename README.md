# NRPyLaTeX
[![Build status](https://www.travis-ci.com/zachetienne/nrpylatex.svg?branch=main)](https://www.travis-ci.com/github/zachetienne/nrpylatex)

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

To install NRPyLaTeX, run the following command

    $ pip install nrpylatex

## Interactive Tutorial (MyBinder)

[Getting Started](https://mybinder.org/v2/gh/zachetienne/nrpytutorial/HEAD?filepath=Tutorial-LaTeX_Parser_Interface.ipynb) | [Guided Example (Cartesian BSSN)](https://mybinder.org/v2/gh/zachetienne/nrpytutorial/HEAD?filepath=Tutorial-LaTeX_Interface_Example-BSSN_Cartesian.ipynb)

## Documentation and Usage

### Simple Example ([Kretschmann Scalar](https://en.wikipedia.org/wiki/Kretschmann_scalar))

        >>> from nrpylatex import parse
        >>> from sympy import simplify
        >>> parse(r"""
        ...     % keydef basis [t, r, \theta, \phi]
        ...     % vardef -zero 'gDD' (4D)
        ...     % vardef -const 'G', 'M'
        ...
        ...     \begin{align}
        ...         %% define Schwarzschild metric
        ...         g_{t t} &= -\left(1 - \frac{2GM}{r}\right) \\
        ...         g_{r r} &=  \left(1 - \frac{2GM}{r}\right)^{-1} \\
        ...         g_{\theta \theta} &= r^{{2}} \\
        ...         g_{\phi \phi} &= r^{{2}} \sin^2\theta
        ...     \end{align}
        ...     % assign -metric 'gDD'
        ...
        ...     \begin{align}
        ...         R^\alpha{}_{\beta \mu \nu} &= \partial_\mu \Gamma^\alpha_{\beta \nu} - \partial_\nu \Gamma^\alpha_{\beta \mu}
        ...             + \Gamma^\alpha_{\mu \gamma} \Gamma^\gamma_{\beta \nu} - \Gamma^\alpha_{\nu \sigma} \Gamma^\sigma_{\beta \mu} \\
        ...         K &= R^{\alpha \beta \mu \nu} R_{\alpha \beta \mu \nu}
        ...     \end{align}
        ... """);
        >>> print(simplify(K))
        48*G**2*M**2/r**6
        
### Indexing Ambiguity

If you attempt to parse `v^2`, that could be converted into `v**2` (a scalar `v` squared) or `vU[2]` (the third component of a vector `vU`). Furthermore, if you already defined `v` or `vU` using `vardef`, we still cannot distinguish between `v**2` and `vU[2]` since both `v` and `vU` can exist in the namespace simultaneously. Therefore, to differentiate between them, we assume vector indexing and require that you use the notation `v^{{2}}` otherwise. To mitigate the task of changing every `v^2` to `v^{{2}}`, we recommend using `srepl "v^{<1>}" -> "v^<1>", "v^<1>" -> "v^{{<1>}}"`. Likewise, if you need to parse `v_2` into a symbol, we recommend using `srepl "v_{<1>}" -> "v_<1>", "v_<1>" -> "\text{v_<1>}"`. We should remark that using a `text` command will build a compound symbol. Finally, to resolve a more complex indexing ambiguity, reference the `srepl` documentation below or visit the interactive tutorial (see above).

#### PARSE MACRO
    parse - parse an equation without rendering

    USAGE
        parse EQUATION

    EQUATION
        syntax: (tensorial) LaTeX equation

#### SREPL MACRO
    srepl - syntactic string replacement

    USAGE
        srepl [OPTION] RULE, ...

    OPTION
        persist
            apply rule(s) to every subsequent input of the parse() function
            remark: internally generated LaTeX included

    RULE
        syntax: "..." -> "..."
        remark (1): string A and string B are considered equal
            if they are equivalent syntactically (i.e. lexeme, token)
        remark (2): substring can include a capture group
            single token capture group <#>
            continuous capture group <#..>

#### VARDEF MACRO
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

#### KEYDEF MACRO
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

#### ASSIGN MACRO
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

#### IGNORE MACRO
    ignore - remove a substring; equivalent to srepl "..." -> "" (empty replacement)

    USAGE
        ignore SUBSTRING, ...
    
    SUBSTRING
        syntax: "..."
