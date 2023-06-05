from nrpylatex.core.scanner import Scanner, ScannerError
from nrpylatex.core.parser import Parser, ParserError
from nrpylatex.core.generator import Generator, GeneratorError
from nrpylatex.utils.structures import IndexedSymbol, IndexedSymbolError
from nrpylatex.utils.exceptions import NRPyLaTeXError, NamespaceError
from nrpylatex.parse_latex import ParseMagic, parse_latex

def load_ipython_extension(ipython):
    ipython.register_magics(ParseMagic)

__version__ = "1.3"
