from IPython.core.magic import Magics, magics_class, line_cell_magic
import nrpylatex as nl, sympy as sp, re, warnings

@magics_class
class ParseMagic(Magics):
    """ NRPyLaTeX IPython Magic """

    @line_cell_magic
    def parse_latex(self, line, cell=None):
        match, kwargs = re.match(r'\s*--([^\s]+)\s*', line), []
        while match:
            kwargs.append(match.group(1))
            line = line[match.span()[-1]:]
            match = re.match(r'\s*--([^\s]+)\s*', line)
        debug = verbose = ignore_warning = False
        for arg in kwargs:
            if arg == 'reset':
                nl.Parser.initialize(reset=True)
            elif arg == 'debug':
                debug = True
            elif arg == 'verbose':
                verbose = True
            elif arg == 'ignore-warning':
                ignore_warning = True
        try:
            sentence = line if cell is None else cell
            state = tuple(nl.Parser._namespace.keys())
            namespace = nl.Parser(debug, verbose).parse_latex(sentence)
            if not isinstance(namespace, dict):
                return namespace
            for key in namespace:
                if isinstance(namespace[key], nl.Tensor):
                    self.shell.user_ns[key] = namespace[key].structure
                elif isinstance(namespace[key], sp.Function('Constant')):
                    self.shell.user_ns[key] = namespace[key].args[0]
            if not namespace: return None
            if ignore_warning:
                return ParseOutput((namespace.values() if verbose else namespace.keys()), sentence)
            overridden = [key for key in state if key in namespace]
            if len(overridden) > 0:
                warnings.warn('some variable(s) in the namespace were overridden', nl.OverrideWarning)
            if verbose:
                for symbol in namespace:
                    if namespace[symbol].symbol in overridden:
                        namespace[symbol].overridden = True
                return ParseOutput(namespace.values(), sentence)
            return ParseOutput((('*' if symbol in overridden else '') + str(symbol)
                for symbol in namespace.keys()), sentence)
        except (nl.ParseError, nl.TensorError) as e:
            print(type(e).__name__ + ': ' + str(e))

class ParseOutput(tuple):
    """ Output Structure for IPython (Jupyter) """

    # pylint: disable = super-init-not-called
    def __init__(self, iterable, sentence):
        self.iterable = iterable
        self.sentence = sentence

    # pylint: disable = unused-argument
    def __new__(cls, iterable, sentence):
        return super(ParseOutput, cls).__new__(cls, iterable)

    def __eq__(self, other):
        return self.iterable == other.iterable and \
            self.sentence == other.sentence

    def __ne__(self, other):
        return self.iterable != other.iterable and \
            self.sentence != other.sentence

    def _repr_latex_(self):
        return r'\[' + self.sentence + r'\]'
