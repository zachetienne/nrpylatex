# NRPyLaTeX

[![CI](https://github.com/zachetienne/nrpylatex/actions/workflows/main.yaml/badge.svg)](https://github.com/zachetienne/nrpylatex/actions/workflows/main.yaml)
[![pypi version](https://img.shields.io/pypi/v/nrpylatex.svg)](https://pypi.org/project/nrpylatex/)
[![Binder](https://mybinder.org/badge.svg)](https://mybinder.org/v2/gh/zachetienne/nrpytutorial/HEAD?filepath=Tutorial-SymPy_LaTeX_Interface.ipynb)
[![Language grade: Python](https://img.shields.io/lgtm/grade/python/g/zachetienne/nrpylatex.svg?logo=lgtm&logoWidth=18)](https://lgtm.com/projects/g/zachetienne/nrpylatex/context:python)

[NRPy+](https://github.com/zachetienne/nrpytutorial)'s LaTeX Interface to SymPy (CAS) for General Relativity

- automatic expansion of
  - [Einstein summation convention](https://en.wikipedia.org/wiki/Einstein_notation)
  - Levi-Civita and Christoffel symbols
  - Lie and covariant derivatives
  - metric inverses and determinants
- automatic index manipulation
- arbitrary coordinate system (default)
- robust exception handling

## &#167; Alternate CAS (Computer Algebra System)

If you are using Mathematica instead of SymPy,

    from sympy import mathematica_code
    
    namespace = parse_latex(...)
    for var in namespace:
        exec(f'{var} = mathematica_code({var})')

If you are using a different CAS, reference the SymPy [documentation](https://docs.sympy.org/latest/modules/printing.html) to find the relevant printing function.

## &#167; Installation

To install **NRPyLaTeX** using [PyPI](https://pypi.org/project/nrpylatex/), run the following command

    $ pip install nrpylatex

## &#167; Interactive Tutorial (MyBinder)

[Getting Started](https://mybinder.org/v2/gh/zachetienne/nrpytutorial/HEAD?filepath=Tutorial-SymPy_LaTeX_Interface.ipynb) | [Guided Example (Cartesian BSSN)](https://mybinder.org/v2/gh/zachetienne/nrpytutorial/HEAD?filepath=Tutorial-LaTeX_Interface_Example-BSSN_Cartesian.ipynb)

## &#167; Documentation and Usage

### Simple Example ([Kretschmann Scalar](https://en.wikipedia.org/wiki/Kretschmann_scalar))

**Python REPL or Script (*.py)**

    >>> from nrpylatex import parse_latex
    >>> parse_latex(r"""
    ...     \begin{align}
    ...         % keydef basis [t, r, \theta, \phi]
    ...         % vardef -zero 'gDD' (4D)
    ...         % vardef -const 'G', 'M'
    ...         %% define Schwarzschild metric diagonal
    ...         g_{t t} &= -\left(1 - \frac{2GM}{r}\right) \\
    ...         g_{r r} &=  \left(1 - \frac{2GM}{r}\right)^{-1} \\
    ...         g_{\theta \theta} &= r^2 \\
    ...         g_{\phi \phi} &= r^2 \sin^2\theta \\
    ...         %% generate metric inverse gUU, determinant det(gDD), and connection GammaUDD
    ...         % assign -metric 'gDD'
    ...         R^\alpha{}_{\beta \mu \nu} &= \partial_\mu \Gamma^\alpha_{\beta \nu} - \partial_\nu \Gamma^\alpha_{\beta \mu}
    ...             + \Gamma^\alpha_{\mu \gamma} \Gamma^\gamma_{\beta \nu} - \Gamma^\alpha_{\nu \sigma} \Gamma^\sigma_{\beta \mu} \\
    ...         K &= R^{\alpha \beta \mu \nu} R_{\alpha \beta \mu \nu}
    ...     \end{align}
    ... """, ignore_warning=True)
    ('gDD', 'epsilonUUUU', 'gdet', 'gUU', 'GammaUDD', 'RUDDD', 'RUUUU', 'RDDDD', 'K')
    >>> from sympy import simplify
    >>> print(simplify(K))
    48*G**2*M**2/r**6

**IPython REPL or Jupyter Notebook**

    In [1]: %load_ext nrpylatex.extension
    In [2]: %%parse_latex --ignore-warning
       ...: \begin{align}
       ...:     % keydef basis [t, r, \theta, \phi]
       ...:     % vardef -zero 'gDD' (4D)
       ...:     % vardef -const 'G', 'M'
       ...:     %% define Schwarzschild metric diagonal
       ...:     g_{t t} &= -\left(1 - \frac{2GM}{r}\right) \\
       ...:     g_{r r} &=  \left(1 - \frac{2GM}{r}\right)^{-1} \\
       ...:     g_{\theta \theta} &= r^2 \\
       ...:     g_{\phi \phi} &= r^2 \sin^2\theta \\
       ...:     %% generate metric inverse gUU, determinant det(gDD), and connection GammaUDD
       ...:     % assign -metric 'gDD'
       ...:     R^\alpha{}_{\beta \mu \nu} &= \partial_\mu \Gamma^\alpha_{\beta \nu} - \partial_\nu \Gamma^\alpha_{\beta \mu}
       ...:         + \Gamma^\alpha_{\mu \gamma} \Gamma^\gamma_{\beta \nu} - \Gamma^\alpha_{\nu \sigma} \Gamma^\sigma_{\beta \mu} \\
       ...:     K &= R^{\alpha \beta \mu \nu} R_{\alpha \beta \mu \nu}
       ...: \end{align}
    Out[2]: ('gDD', 'epsilonUUUU', 'gdet', 'gUU', 'GammaUDD', 'RUDDD', 'RUUUU', 'RDDDD', 'K')
    In [3]: from sympy import simplify
    In [4]: print(simplify(K))
    Out[4]: 48*G**2*M**2/r**6

### Einstein Summation Convention

If the same index appears exactly twice in any single term, assume summation over the range of that index.

    M^{n k} v_k := \sum_k M^{n k} v_k = M^{n 0} v_0 + M^{n 1} v_1 + ...

### Indexing Ambiguity

If `v` and `vU` are both in the namespace and you attempt to parse the expression `v^2`, the output from **NRPyLaTeX** will be `vU[2]` (the third component of `vU`). To resolve the indexing ambiguity between `vU[2]` and `v**2` (`v` squared), we suggest using the notation `v^{{2}}` since `v^2` and `v^{{2}}` are rendered identically in LaTeX. Similarly, if `v` and `vD` are both in the namespace and you are parsing the symbol `v_2`, we suggest replacing `v_2` with `\text{v_2}` using the `srepl` macro to preserve the LaTeX rendering and build a compound symbol.

    srepl "v_{<1..>}" -> "v<>{<1..>}", "v_<1>" -> "\text{v_<1>}", "v<>{<1..>}" -> "\text{v_<1..>}"

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

    OPERAND
        basis=BASIS
            desc: define basis (or coordinate system)
            example(s): [x, y, z], [r, \phi], default

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
