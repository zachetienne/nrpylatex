# NRPyLaTeX Documentation
[![arXiv](https://img.shields.io/badge/arXiv-2111.05861-B31B1B)](https://arxiv.org/abs/2111.05861)

NRPyLaTeX is a frontend interface for [SymPy](https://www.sympy.org/en/index.html), an open-source computer algebra package for Python, that enables the input of tensor equations, expressed in [Einstein notation](https://en.wikipedia.org/wiki/Einstein_notation), using [LaTeX](https://en.wikipedia.org/wiki/LaTeX), a standard typesetting system for mathematics. In addition to SymPy, NRPyLaTeX also supports output to Mathematica and Maple using the SymPy [printing](https://docs.sympy.org/latest/modules/printing.html) system. The principal motivation for developing NRPyLaTeX was to reduce the learning curve for inputting and manipulating tensor equations in computer algebra systems, which typically use their own unique notation. However, LaTeX is only a typesetting system, and as such is not designed to resolve ambiguities that may arise in mathematical expressions. To address this, NRPyLaTeX implements a convenient command system that, e.g., defines variables with specific attributes. Furthermore, the entire command system is embedded inside of LaTeX comments, enabling the seamless integration of entire NRPyLaTeX workflows into the LaTeX source code of scientific papers without interfering with the rendered document. Additionally, NRPyLaTeX adopts [NRPy+](https://github.com/zachetienne/nrpytutorial)'s tensor notation, enabling NRPyLaTeX output to be directly converted into highly optimized C code using NRPy+. Finally, NRPyLaTeX implements a robust error handling system that identifies common indexing errors and unresolved ambiguities with Einstein notation.

## Installation

To install **NRPyLaTeX** using [PyPI](https://pypi.org/project/nrpylatex/), run the following command in the terminal

    $ pip install nrpylatex

## Exporting (CAS)

If you are using [Mathematica](https://www.wolfram.com/mathematica/) (or [Maple](https://www.maplesoft.com/products/Maple/)) instead of SymPy, run the following code to convert your output

    from sympy import mathematica_code
    
    namespace = parse_latex(...)
    for var in namespace:
        exec(f'{var} = mathematica_code({var})')

If you are using a different CAS, reference the SymPy [documentation](https://docs.sympy.org/latest/modules/printing.html) to find the relevant printing function.
