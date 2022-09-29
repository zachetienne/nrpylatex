""" NRPyLaTeX: Convert LaTeX Sentence to SymPy Expression """
# Author: Ken Sible
# Email:  ksible *at* outlook *dot* com

from nrpylatex.core.parser import Parser, Tensor, OverrideWarning
from sympy import Function, Symbol
from inspect import currentframe
import warnings

def parse_latex(sentence, reset=False, debug=False, namespace=None, ignore_warning=False):
    """ Convert LaTeX Sentence to SymPy Expression

        :arg: latex sentence (raw string)
        :arg: reset namespace
        :arg: debug parse_latex()
        :arg: import namespace (dict)
        :arg: ignore OverrideWarning
        :return: namespace diff or expression
    """
    if reset: Parser.initialize(reset=True)
    if namespace:
        for symbol in namespace:
            function = Function('Tensor')(Symbol(symbol, real=True))
            structure = namespace[symbol]
            if not isinstance(structure, list):
                raise ImportError('cannot import variable of type %s, only list' % type(structure))
            dimension = len(structure)
            i = 0
            while isinstance(structure[i], list):
                if len(structure[i] != dimension):
                    raise ImportError('inconsistent dimension in \'%s\'' % symbol)
                i += 1
            Parser._namespace[symbol] = Tensor(function, dimension, structure)

    state = tuple(Parser._namespace.keys())
    namespace = Parser(debug).parse_latex(sentence)
    if not isinstance(namespace, dict):
        return namespace
    if not namespace: return None

    frame = currentframe().f_back
    for key in namespace:
        if isinstance(namespace[key], Tensor):
            frame.f_globals[key] = namespace[key].structure
        elif isinstance(namespace[key], Function('Constant')):
            frame.f_globals[key] = namespace[key].args[0]

    if ignore_warning:
        return tuple(namespace.keys())
    overridden = [key for key in state if key in namespace]
    if len(overridden) > 0:
        warnings.warn('some variable(s) in the namespace were overridden', OverrideWarning)
    return tuple(('*' if symbol in overridden else '')
        + str(symbol) for symbol in namespace.keys())

class ImportError(Exception):
    """ Illegal Namespace Import """
