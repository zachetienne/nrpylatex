![NRPyLaTeX Logo](https://raw.githubusercontent.com/zachetienne/nrpylatex/main/docs/imgs/logo.png)

---

[![CI](https://github.com/zachetienne/nrpylatex/actions/workflows/main.yaml/badge.svg)](https://github.com/zachetienne/nrpylatex/actions/workflows/main.yaml)
[![PyPI](https://img.shields.io/pypi/v/nrpylatex.svg)](https://pypi.org/project/nrpylatex/)
[![Binder](https://mybinder.org/badge_logo.svg)](https://mybinder.org/v2/gh/zachetienne/nrpylatex.git/HEAD?filepath=docs%2FNRPyLaTeX%20Tutorial.ipynb)
[![arXiv](https://img.shields.io/badge/arXiv-2111.05861-B31B1B)](https://arxiv.org/abs/2111.05861)

[NRPy+](https://github.com/zachetienne/nrpytutorial)'s LaTeX Interface to SymPy (CAS) for General Relativity

- automatic expansion of
  - [Einstein summation convention](https://en.wikipedia.org/wiki/Einstein_notation)
  - Levi-Civita and Christoffel symbols
  - Lie and covariant derivatives
  - metric inverse and determinant
- automatic index raising and lowering
- arbitrary coordinate system (default)
- exception handling and debugging

## &#167; Installation

To install **NRPyLaTeX** using [PyPI](https://pypi.org/project/nrpylatex/), run the following command in the terminal

    $ pip install nrpylatex

## &#167; Exporting (CAS)

If you are using Mathematica instead of SymPy, run the following code to convert your output

    from sympy import mathematica_code
    
    namespace = parse_latex(...)
    for var in namespace:
        exec(f'{var} = mathematica_code({var})')

If you are using a different CAS, reference the SymPy [documentation](https://docs.sympy.org/latest/modules/printing.html) to find the relevant printing function.

## &#167; Interactive Tutorial (MyBinder)

[Quick Start](https://mybinder.org/v2/gh/zachetienne/nrpylatex.git/HEAD?filepath=docs%2FNRPyLaTeX%20Tutorial.ipynb) | [NRPy+ Integration](https://mybinder.org/v2/gh/zachetienne/nrpytutorial/HEAD?filepath=Tutorial-SymPy_LaTeX_Interface.ipynb) | [Guided Example (BSSN Formalism)](https://mybinder.org/v2/gh/zachetienne/nrpytutorial/HEAD?filepath=Tutorial-LaTeX_Interface_Example-BSSN_Cartesian.ipynb)

## &#167; Documentation and Usage

[Getting Started and API Reference](https://zachetienne.github.io/nrpylatex/)

### Simple Example ([Kretschmann Scalar](https://en.wikipedia.org/wiki/Kretschmann_scalar))

**Python REPL or Script (*.py)**

    >>> from nrpylatex import parse_latex
    >>> parse_latex(r"""
    ...     % ignore "\begin{align}" "\end{align}"
    ...     \begin{align}
    ...         % coord [t, r, \theta, \phi]
    ...         % define gDD --dim 4 --zeros
    ...         % define G M --const
    ...         %% define Schwarzschild metric diagonal
    ...         g_{t t} &= -\left(1 - \frac{2GM}{r}\right) \\
    ...         g_{r r} &=  \left(1 - \frac{2GM}{r}\right)^{-1} \\
    ...         g_{\theta \theta} &= r^2 \\
    ...         g_{\phi \phi} &= r^2 \sin^2{\theta} \\
    ...         %% generate metric inverse gUU, determinant det(gDD), and connection GammaUDD
    ...         % assign gDD --metric
    ...         R^\alpha{}_{\beta \mu \nu} &= \partial_\mu \Gamma^\alpha_{\beta \nu} - \partial_\nu \Gamma^\alpha_{\beta \mu}
    ...             + \Gamma^\alpha_{\mu \gamma} \Gamma^\gamma_{\beta \nu} - \Gamma^\alpha_{\nu \sigma} \Gamma^\sigma_{\beta \mu} \\
    ...         K &= R^{\alpha \beta \mu \nu} R_{\alpha \beta \mu \nu}
    ...     \end{align}
    ... """)
    ('G', 'GammaUDD', 'gDD', 'gUU', 'epsilonUUUU', 'RUDDD', 'K', 'RUUUU', 'M', 'r', 'theta', 'RDDDD', 'gdet')
    >>> from sympy import simplify
    >>> print(simplify(K))
    48*G**2*M**2/r**6

**IPython REPL or Jupyter Notebook**

    In [1]: %load_ext nrpylatex
    In [2]: %%parse_latex
       ...: % ignore "\begin{align}" "\end{align}"
       ...: \begin{align}
       ...:     % coord [t, r, \theta, \phi]
       ...:     % define gDD --dim 4 --zeros
       ...:     % define G M --const
       ...:     %% define Schwarzschild metric diagonal
       ...:     g_{t t} &= -\left(1 - \frac{2GM}{r}\right) \\
       ...:     g_{r r} &=  \left(1 - \frac{2GM}{r}\right)^{-1} \\
       ...:     g_{\theta \theta} &= r^2 \\
       ...:     g_{\phi \phi} &= r^2 \sin^2{\theta} \\
       ...:     %% generate metric inverse gUU, determinant det(gDD), and connection GammaUDD
       ...:     % assign gDD --metric
       ...:     R^\alpha{}_{\beta \mu \nu} &= \partial_\mu \Gamma^\alpha_{\beta \nu} - \partial_\nu \Gamma^\alpha_{\beta \mu}
       ...:         + \Gamma^\alpha_{\mu \gamma} \Gamma^\gamma_{\beta \nu} - \Gamma^\alpha_{\nu \sigma} \Gamma^\sigma_{\beta \mu} \\
       ...:     K &= R^{\alpha \beta \mu \nu} R_{\alpha \beta \mu \nu}
       ...: \end{align}
    Out[2]: ('G', 'GammaUDD', 'gDD', 'gUU', 'epsilonUUUU', 'RUDDD', 'K', 'RUUUU', 'M', 'r', 'theta', 'RDDDD', 'gdet')
    In [3]: from sympy import simplify
    In [4]: print(simplify(K))
    Out[4]: 48*G**2*M**2/r**6
