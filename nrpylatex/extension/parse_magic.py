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
        verbose = ignore_warning = False
        for arg in kwargs:
            if arg == 'reset':
                nl.Parser.initialize(reset=True)
            elif arg == 'verbose':
                verbose = True
            elif arg == 'ignore-warning':
                ignore_warning = True
        action = 'ignore' if ignore_warning else 'default'
        warnings.filterwarnings(action, category=nl.OverrideWarning)
        try:
            sentence = line if cell is None else cell
            duplicate_namespace = nl.Parser._namespace.copy()
            namespace = nl.Parser(verbose).parse_latex(sentence)
            if not isinstance(namespace, dict):
                return namespace
            key_diff = [key for key in namespace if key not in duplicate_namespace]
            for key in namespace:
                if isinstance(namespace[key], nl.Tensor):
                    tensor = namespace[key]
                    if not tensor.equation and tensor.rank == 0:
                        if key in key_diff:
                            key_diff.remove(key)
                    self.shell.user_ns[key] = namespace[key].structure
                elif isinstance(namespace[key], sp.Function('Constant')):
                    if key in key_diff:
                        key_diff.remove(key)
                    self.shell.user_ns[key] = namespace[key].args[0]
            return ParseOutput(key_diff if not verbose \
                else [namespace[key] for key in key_diff], sentence)
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
