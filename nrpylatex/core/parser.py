""" NRPyLaTeX Parser """
# Author: Ken Sible
# Email:  ksible *at* outlook *dot* com

from nrpylatex.core.scanner import Scanner
from nrpylatex.core.generator import Generator
from nrpylatex.utils.structures import ExprTree, CoordinateSystem, symdef
from nrpylatex.utils.structures import IndexedSymbol, IndexedSymbolError
from nrpylatex.utils.exceptions import NRPyLaTeXError
from nrpylatex.utils.functional import product
from sympy import Function, Derivative, Symbol, Integer, Rational, Float, Pow
from sympy import sin, cos, tan, sinh, cosh, tanh, asin, acos, atan, asinh, acosh, atanh
from sympy import pi, exp, log, sqrt, expand, diff
from collections import OrderedDict
import re

class Parser:

    _namespace, _property = OrderedDict(), {}

    def __init__(self, debug=False):
        self.scanner = Scanner()
        self.generator = Generator(self)
        self.state = OrderedDict()
        if not self._property:
            self.initialize()
        self._property['debug'] = int(debug)
        for symbol in self._namespace:
            self._namespace[symbol].overridden = False

    @staticmethod
    def initialize(reset=False):
        if reset: Parser._namespace.clear()
        Parser._property['srepl'] = []
        Parser._property['coord'] = CoordinateSystem('x')
        Parser._property['index'] = {chr(i): 3 for i in range(97, 123)}
        Parser._property['index'].update({i: 4 for i in [
            'alpha', 'beta', 'gamma', 'delta', 'epsilon', 'zeta', 'eta', 'theta',
            'iota', 'kappa', 'lambda', 'mu', 'nu', 'xi', 'omikron', 'pi', 'rho',
            'sigma', 'tau', 'upsilon', 'phi', 'chi', 'psi', 'omega'
        ]})
        Parser._property['ignore'] = ['\\left', '\\right', '{}', '&']
        Parser._property['metric'] = {'': 'g', 'bar': 'g', 'hat': 'g', 'tilde': 'gamma'}
        Parser._property['suffix'] = None

    def parse_latex(self, sentence):
        """ Parse LaTeX Sentence

            :arg:    latex sentence (raw string)
            :return: namespace or expression
        """
        # replace every substring marked 'ignore' with an empty string
        for ignore in self._property['ignore']:
            sentence = sentence.replace(ignore, '')
        # perform string replacement (aliasing) using namespace mapping
        self.scanner.initialize('\n'.join(['srepl "%s" -> "%s"' % (old, new)
            for (old, new) in self._property['srepl']] + [sentence]))
        self.scanner.lex()
        for _ in self._property['srepl']:
            self._srepl()
        sentence = self.scanner.sentence[self.scanner.mark():]
        stack = []; i = i_1 = i_2 = i_3 = 0
        while i < len(sentence):
            lexeme = sentence[i]
            if   lexeme == '(': stack.append(i)
            elif lexeme == ')':
                i_1, i_2 = stack.pop(), i + 1
                if i_2 < len(sentence) and sentence[i_2] != '_':
                    i_1 = i_2 = 0
            elif i_2 != 0:
                # replace comma notation with operator notation for parenthetical expression(s)
                if lexeme == ',' and sentence[i - 1] == '{':
                    i_3 = sentence.find('}', i) + 1
                    subexpr, indexing = sentence[i_1:i_2], sentence[i_2:i_3][3:-1]
                    indexing = reversed(re.findall(self.scanner.token_dict['LETTER'], indexing))
                    operator = ' '.join('\\partial_{' + index + '}' for index in indexing)
                    sentence = sentence.replace(sentence[i_1:i_3], operator + ' ' + subexpr)
                    i = i_1 + len(operator + ' ' + subexpr) - 1
                # replace semicolon notation with operator notation for parenthetical expression(s)
                elif lexeme == ';' and sentence[i - 1] == '{':
                    i_3 = sentence.find('}', i) + 1
                    subexpr, indexing = sentence[i_1:i_2], sentence[i_2:i_3][3:-1]
                    indexing = reversed(re.findall(self.scanner.token_dict['LETTER'], indexing))
                    operator = ' '.join('\\nabla_{' + index + '}' for index in indexing)
                    sentence = sentence.replace(sentence[i_1:i_3], operator + ' ' + subexpr)
                    i = i_1 + len(operator + ' ' + subexpr) - 1
            i += 1
        i = 0
        # replace every comment (%%...\n) with an empty string
        while i < len(sentence) - 1:
            if sentence[i:(i + 2)] == '%%':
                index = sentence.index('\n', i + 2)
                sentence = sentence.replace(sentence[i:index], '')
            else: i += 1
        self.scanner.initialize(sentence)
        self.scanner.lex()
        expression = self._latex()
        if expression is not None:
            return expression
        return {symbol: self._namespace[symbol] for symbol in self.state}

    # <LATEX> -> ( '%' <MACRO> | [ '%' ] <ASSIGNMENT> ) { ( '%' <MACRO> | [ '%' ] <ASSIGNMENT> ) }*
    def _latex(self):
        count = 0
        while self.scanner.lexeme:
            if self.accept('COMMENT'):
                if any(self.peek(token) for token in ('DEFINE_MACRO', 'ASSIGN_MACRO',
                        'IGNORE_MACRO', 'SREPL_MACRO', 'COORD_MACRO', 'INDEX_MACRO')):
                    self._macro()
                else: self._assignment()
            elif count > 0:
                self._assignment()
            else:
                if any(self.peek(token) for token in ('PAR_SYM', 'COV_SYM', 'LIE_SYM', 'DIACRITIC', 'SYMB_CMD')) \
                        or (self.peek('LETTER') and self.scanner.lexeme != 'e'):
                    marker = self.scanner.mark()
                    self._operator('LHS')
                    assignment = self.accept('EQUAL')
                    self.scanner.reset(marker)
                else: assignment = False
                if assignment:
                    self._assignment()
                else:
                    tree = ExprTree(self._expression())
                    for subtree in tree.preorder():
                        subexpr, rank = subtree.expr, len(subtree.expr.args)
                        if rank == 1 and subexpr.func == Function('Tensor'):
                            subtree.expr = subexpr.args[0]
                            del subtree.children[:]
                    return tree.reconstruct()
            count += 1
        return None

    # <MACRO> -> <DEFINE> | <ASSIGN> | <IGNORE> | <SREPL> | <COORD> | <INDEX>
    def _macro(self):
        macro = self.scanner.lexeme
        if self.peek('DEFINE_MACRO'):
            self._define()
        elif self.peek('ASSIGN_MACRO'):
            self._assign()
        elif self.peek('IGNORE_MACRO'):
            self._ignore()
        elif self.peek('SREPL_MACRO'):
            self._srepl()
        elif self.peek('COORD_MACRO'):
            self._coord()
        elif self.peek('INDEX_MACRO'):
            self._index()
        else:
            sentence, position = self.scanner.sentence, self.scanner.mark()
            raise ParserError('unsupported macro \'%s\' at position %d' %
                (macro, position), sentence, position)

    # <DEFINE> -> <DEFINE_MACRO> { <VARIABLE> }+ { '--' ( <CONST> | <ZEROS> | <OPTION> ) }*
    def _define(self):
        self.scanner.whitespace = True
        self.expect('DEFINE_MACRO')
        symbols = []
        while True:
            self.accept('WHITESPACE')
            symbols.append(self._variable())
            self.accept('WHITESPACE')
            if self.peek('MINUS') or self.peek('LINEBREAK'): break
        dimension = suffix = symmetry = metric = weight = None
        const = zeros = False
        while self.accept('MINUS'):
            self.expect('MINUS')
            self.accept('WHITESPACE')
            const = self.accept('CONST')
            self.accept('WHITESPACE')
            zeros = self.accept('ZEROS')
            self.accept('WHITESPACE')
            if not (const or zeros):
                option, value = self._option().split('<>')
                if option == 'dimension':
                    dimension = int(value)
                elif option == 'suffix':
                    if suffix != 'none':
                        suffix = value
                elif option == 'symmetry':
                    symmetry = value
                elif option == 'metric':
                    metric = value
                elif option == 'weight':
                    weight = value
            self.accept('WHITESPACE')
        for symbol in symbols:
            if const:
                self._namespace[symbol] = Function('Constant')(Symbol(symbol, real=True))
                self.state[symbol] = None
            else:
                function = Function('Tensor')(Symbol(symbol, real=True))
                tensor = IndexedSymbol(function, dimension, suffix=suffix, metric=metric, weight=weight)
                tensor.symmetry = 'sym01' if symmetry == 'metric' else symmetry
                self._define_tensor(tensor, zeros=zeros)
                if symmetry == 'metric':
                    diacritic = next(diacritic for diacritic in ('bar', 'hat', 'tilde', '') if diacritic in symbol)
                    self._property['metric'][diacritic] = re.split(diacritic if diacritic else r'[UD]', symbol)[0]
                    with self.scanner.new_context():
                        self.scanner.whitespace = False
                        self.accept('WHITESPACE')
                        self.accept('LINEBREAK')
                        self.parse_latex(Generator.generate_metric(symbol, dimension, diacritic, suffix))
        self.scanner.whitespace = False
        self.accept('WHITESPACE')
        self.accept('LINEBREAK')

    # <ASSIGN> -> <ASSIGN_MACRO> { <VARIABLE> }+ { '--' <OPTION> }+
    def _assign(self):
        self.scanner.whitespace = True
        self.expect('ASSIGN_MACRO')
        symbols = []
        while True:
            self.accept('WHITESPACE')
            symbols.append(self._variable())
            self.accept('WHITESPACE')
            if self.peek('MINUS') or self.peek('LINEBREAK'): break
        dimension = suffix = symmetry = metric = weight = None
        while self.accept('MINUS'):
            self.expect('MINUS')
            self.accept('WHITESPACE')
            option, value = self._option().split('<>')
            if option == 'dimension':
                dimension = int(value)
            elif option == 'suffix':
                if suffix != 'none':
                    suffix = value
            elif option == 'symmetry':
                symmetry = value
            elif option == 'metric':
                metric = value
            elif option == 'weight':
                weight = value
            self.accept('WHITESPACE')
        for symbol in symbols:
            if symbol not in self._namespace:
                raise IndexedSymbolError('cannot assign attribute(s) to undefined variable \'{symbol}\''.format(symbol=symbol))
            tensor = self._namespace[symbol]
            if dimension:
                if tensor.rank > 0:
                    raise IndexedSymbolError('cannot assign dimension to \'{symbol}\' since rank({symbol}) > 0'.format(symbol=symbol))
                tensor.dimension = dimension
            if suffix:
                tensor.suffix = suffix
            if symmetry:
                tensor.symmetry = 'sym01' if symmetry == 'metric' else symmetry
            if metric:
                tensor.metric = metric
            if weight:
                tensor.weight = weight
            if symmetry == 'metric':
                if symbol not in self._namespace:
                    raise IndexedSymbolError('cannot assign --metric to undefined variable \'{symbol}\''.format(symbol=symbol))
                tensor = self._namespace[symbol]
                if tensor.rank != 2:
                    raise IndexedSymbolError('cannot assign --metric to \'{symbol}\' since rank({symbol}) != 2'.format(symbol=symbol))
                structure = tensor.structure
                for i in range(tensor.dimension):
                    for j in range(tensor.dimension):
                        if structure[i][j] == 0:
                            structure[i][j] = structure[j][i]
                        elif structure[j][i] == 0:
                            structure[j][i] = structure[i][j]
                        if structure[i][j] != structure[j][i]:
                            raise IndexedSymbolError('cannot assign --metric to \'{symbol}\' since {symbol}[{i}][{j}] != {symbol}[{j}][{i}]'
                                .format(symbol=symbol, i=i, j=j))
                diacritic = next(diacritic for diacritic in ('bar', 'hat', 'tilde', '') if diacritic in symbol)
                self._property['metric'][diacritic] = re.split(diacritic if diacritic else r'[UD]', symbol)[0]
                with self.scanner.new_context():
                    self.scanner.whitespace = False
                    self.accept('WHITESPACE')
                    self.accept('LINEBREAK')
                    self.parse_latex(Generator.generate_metric(symbol, tensor.dimension, diacritic, tensor.suffix))
            base_symbol = re.split(r'_d|_dup|_cd|_ld', symbol)[0]
            if base_symbol and tensor.suffix:
                rank = 0
                for rank_symbol in re.split(r'_d|_dup|_cd|_ld', symbol):
                    for character in reversed(rank_symbol):
                        if character in ('U', 'D'):
                            rank += 1
                        else: break
                if base_symbol in self._namespace:
                    self._namespace[base_symbol].suffix = tensor.suffix
                elif rank == 0:
                    function = Function('Tensor')(Symbol(base_symbol, real=True))
                    self._define_tensor(IndexedSymbol(function, suffix=tensor.suffix))
        self.scanner.whitespace = False
        self.accept('WHITESPACE')
        self.accept('LINEBREAK')

    # <IGNORE> -> <IGNORE_MACRO> { <STRING> }+
    def _ignore(self):
        self.expect('IGNORE_MACRO')
        while True:
            string = self.scanner.lexeme[1:-1]
            if len(string) > 0 and string not in self._property['ignore']:
                self._property['ignore'].append(string)
            sentence, position = self.scanner.sentence, self.scanner.index
            self.scanner.mark()
            self.expect('STRING')
            if len(string) > 0:
                self.scanner.sentence = sentence[:position] + sentence[position:].replace(string, '')
            if not self.peek('STRING'): break
        self.scanner.reset(); self.scanner.lex()

    # <SREPL> -> <SREPL_MACRO> <STRING> '->' <STRING> [ '--' <PERSIST> ]
    def _srepl(self):
        self.expect('SREPL_MACRO')
        old = self.scanner.lexeme[1:-1]
        self.expect('STRING')
        self.expect('ARROW')
        new = self.scanner.lexeme[1:-1]
        self.scanner.mark()
        self.expect('STRING')
        persist = False
        if self.accept('MINUS'):
            self.expect('MINUS')
            self.scanner.mark()
            self.expect('PERSIST')
            persist = True
        if persist and [old, new] not in self._property['srepl']:
            self._property['srepl'].append([old, new])
        self.scanner.reset(); self.scanner.mark()
        scanner = Scanner(); scanner.initialize(old)
        substr_syntax = []
        for token in scanner.tokenize():
            substr_syntax.append((scanner.lexeme, token))
        string_syntax = []
        for token in self.scanner.tokenize():
            string_syntax.append((self.scanner.index, self.scanner.lexeme, token))
        sentence = self.scanner.sentence
        i_1 = i_2 = offset = 0
        for i, (index, lexeme, token) in enumerate(string_syntax):
            if substr_syntax[0][0] == lexeme or substr_syntax[0][1] == 'GROUP' or token == 'SYMB_CMD':
                k, index, varmap = i, index - len(lexeme), {}
                for j, (_lexeme, _token) in enumerate(substr_syntax, start=i):
                    if k >= len(string_syntax): break
                    if _token == 'LETTER' and string_syntax[k][2] == 'SYMB_CMD':
                        letter_1 = _lexeme[1:] if len(_lexeme) > 1 else _lexeme
                        letter_2, l = string_syntax[k + 2][1], 2
                        while string_syntax[k + l + 1][2] != 'RBRACE':
                            letter_2 += string_syntax[k + l + 1][1]
                            l += 1
                        if letter_1 != letter_2: break
                        k += l + 1
                    elif _token == 'GROUP':
                        varmap[_lexeme] = string_syntax[k][1]
                        if _lexeme[-2] == '.':
                            l, string = k + 1, varmap[_lexeme]
                            if l < len(string_syntax) and j - i + 1 < len(substr_syntax):
                                EOL = substr_syntax[j - i + 1]
                                while string_syntax[l][1] != EOL[0]:
                                    if EOL[1] == 'LETTER' and string_syntax[l][2] == 'SYMB_CMD':
                                        letter_1 = EOL[0][1:] if len(EOL[0]) > 1 else EOL[0]
                                        letter_2, m = string_syntax[l + 2][1], 2
                                        while string_syntax[l + m + 1][2] != 'RBRACE':
                                            letter_2 += string_syntax[l + m + 1][1]
                                            m += 1
                                        if letter_1 == letter_2:
                                            string_syntax[l + 1] = (string_syntax[l - 1][0], EOL[0], EOL[1])
                                        else:
                                            string += '\\mathrm{' + letter_2 + '}'
                                            l += m + 1
                                    else: string += string_syntax[l][1]
                                    if l + 1 >= len(string_syntax): break
                                    l += 1
                                else:
                                    k, varmap[_lexeme] = l - 1, string
                    elif _lexeme != string_syntax[k][1]: break
                    if (j - i + 1) == len(substr_syntax):
                        new_repl = new
                        for var in varmap:
                            new_repl = new_repl.replace(var, varmap[var])
                        i_1, i_2 = index + offset, string_syntax[k][0] + offset
                        old_repl = sentence[i_1:i_2]
                        sentence = sentence[:i_1] + new_repl + sentence[i_2:]
                        offset += len(new_repl) - len(old_repl)
                    k += 1
        self.scanner.sentence = sentence
        self.scanner.reset(); self.scanner.lex()

    # <COORD> -> <COORD_MACRO> ( '[' <SYMBOL> [ ',' <SYMBOL> ]* ']' | '--' <DEFAULT> )
    def _coord(self):
        self.expect('COORD_MACRO')
        if self.accept('MINUS'):
            self.expect('MINUS')
            self.expect('DEFAULT')
            self._property['coord'] = CoordinateSystem('x')
        else:
            self.expect('LBRACK')
            del self._property['coord'][:]
            while True:
                symbol = self._strip(self._symbol())
                if symbol in self._property['coord']:
                    sentence, position = self.scanner.sentence, self.scanner.mark()
                    raise ParserError('duplicate coord symbol \'%s\' at position %d' %
                        (sentence[position], position), sentence, position)
                self._property['coord'].append(Symbol(symbol, real=True))
                if not self.accept('COMMA'): break
            self.expect('RBRACK')

    # <INDEX> -> <INDEX_MACRO> [ { <LETTER> | '[' <LETTER> '-' <LETTER> ']' }+ | '--' <DEFAULT> ] '--' <DIM> <INTEGER>
    def _index(self):
        self.expect('INDEX_MACRO')
        indices = []
        if self.accept('MINUS'):
            self.expect('MINUS')
            self.expect('DEFAULT')
            indices.extend([index for index in self._property['index']])
        else:
            while True:
                sentence, position = self.scanner.sentence, self.scanner.mark()
                if self.accept('LBRACK'):
                    index_1 = self._strip(self.scanner.lexeme)
                    self.expect('LETTER')
                    self.expect('MINUS')
                    index_2 = self._strip(self.scanner.lexeme)
                    self.expect('LETTER')
                    if len(index_1) > 1 or len(index_2) > 1:
                        raise ParserError('unsupported index range \'%s\' at position %d' %
                            (sentence[position:self.scanner.index], position), sentence, position)
                    indices.extend([chr(i) for i in range(ord(index_1), ord(index_2) + 1)])
                    self.expect('RBRACK')
                elif self.peek('LETTER'):
                    indices.append(self._strip(self.scanner.lexeme))
                    self.expect('LETTER')
                if self.peek('MINUS'): break
        self.expect('MINUS')
        self.expect('MINUS')
        self.expect('DIM')
        dimension = self.scanner.lexeme
        self.expect('INTEGER')
        dimension = int(dimension)
        if self.accept('MINUS'):
            self.expect('MINUS')
            self.expect('DEFAULT')
            indices = [index for index in self._property['index']]
        self._property['index'].update({index: dimension for index in indices})

    # <OPTION> -> <DIM> <INTEGER> | <SYM> <SYMMETRY> | <WEIGHT> <NUMBER> | <SUFFIX> <VARIABLE> | <METRIC> [ <VARIABLE> ]
    def _option(self):
        if self.accept('DIM'):
            self.accept('WHITESPACE')
            dimension = self.scanner.lexeme
            self.expect('INTEGER')
            return 'dimension<>' + dimension
        if self.accept('SYM'):
            self.accept('WHITESPACE')
            symmetry = self.scanner.lexeme
            self.expect('SYMMETRY')
            return 'symmetry<>' + symmetry
        if self.accept('WEIGHT'):
            self.accept('WHITESPACE')
            weight = self._number()
            return 'weight<>' + str(weight)
        if self.accept('SUFFIX'):
            self.accept('WHITESPACE')
            sentence, position = self.scanner.sentence, self.scanner.mark()
            suffix = self._variable()
            if suffix[0] != 'd' or suffix[-1] != 'D':
                raise ParserError('unsupported suffix \'%s\' at position %d' %
                    (suffix, position), sentence, position)
            return 'suffix<>' + suffix
        if self.accept('METRIC'):
            self.accept('WHITESPACE')
            if self.peek('LETTER'):
                metric = self._variable()
                return 'metric<>' + metric
            return 'symmetry<>metric'
        sentence, position = self.scanner.sentence, self.scanner.mark()
        raise ParserError('unexpected \'%s\' at position %d' %
            (sentence[position], position), sentence, position)

    # <ASSIGNMENT> -> <OPERATOR> = <EXPRESSION> [ '\\' ] [ '%' <NOIMPSUM> ]
    def _assignment(self):
        function = self._operator('LHS')
        indexed = function.func == Function('Tensor') and len(function.args) > 1
        self.expect('EQUAL')
        sentence, position_1 = self.scanner.sentence, self.scanner.mark()
        tree = ExprTree(self._expression())
        self.accept('NEWLINE')
        position_2 = self.scanner.mark()
        impsum = True
        if self.accept('COMMENT'):
            if self.accept('NOIMPSUM'):
                impsum = False
            else: self.scanner.reset()
        equation = ((IndexedSymbol.latex_format(function), sentence[position_1:position_2]), tree.root.expr)
        if self._property['debug']:
            (latex_LHS, latex_RHS), expr_RHS = equation
            lineno = '[%d]' % self._property['debug']
            print('%s LaTeX' % lineno)
            print('%s   %s = %s' % (len(lineno) * ' ', latex_LHS, latex_RHS.rstrip()))
            print('%s SymPy' % (len(lineno) * ' '))
            print('%s   %s = %s' % (len(lineno) * ' ', function, expr_RHS))
        if not indexed:
            for subtree in tree.preorder():
                subexpr, rank = subtree.expr, len(subtree.expr.args)
                if subexpr.func == Function('Tensor') and rank > 1:
                    indexed = True
        LHS, RHS = function, expand(tree.root.expr) if indexed else tree.root.expr
        global_env, dimension = self.generator.generate(LHS, RHS, impsum)
        symbol, indices = str(function.args[0]), function.args[1:]
        if any(isinstance(index, Integer) for index in indices):
            tensor = self._namespace[symbol]
            tensor.structure = global_env[symbol]
        else:
            suffix = self._namespace[symbol].suffix if symbol in self._namespace else None
            tensor = IndexedSymbol(function, dimension, structure=global_env[symbol],
                equation=equation, suffix=suffix, impsum=impsum)
        self._namespace[symbol] = tensor
        self.state[symbol] = None

    # <EXPRESSION> -> <TERM> { ( '+' | '-' ) <TERM> }*
    def _expression(self):
        expr = self._term()
        while self.peek('PLUS') or self.peek('MINUS'):
            if self.accept('PLUS'):
                expr += self._term()
            elif self.accept('MINUS'):
                expr -= self._term()
        return expr

    # <TERM> -> <FACTOR> { [ '/' ] <FACTOR> }*
    def _term(self):
        expr = self._factor()
        while any(self.peek(token) for token in ('DIVIDE',
                'RATIONAL', 'DECIMAL', 'INTEGER', 'PI', 'PAR_SYM', 'COV_SYM', 'LIE_SYM',
                'SYMB_CMD', 'FUNC_CMD', 'FRAC_CMD', 'SQRT_CMD', 'NLOG_CMD', 'TRIG_CMD',
                'LPAREN', 'LBRACK', 'DIACRITIC', 'LETTER', 'COMMAND', 'COMMENT', 'BACKSLASH')):
            self.scanner.mark()
            if self.accept('COMMENT'):
                if not self.peek('SUFFIX'):
                    self.scanner.reset()
                    return expr
                self.scanner.reset()
            if self.accept('BACKSLASH'):
                if self.peek('RBRACE'):
                    self.scanner.reset()
                    return expr
                self.scanner.reset()
            if self.accept('DIVIDE'):
                expr /= self._factor()
            else: expr *= self._factor()
        return expr

    # <FACTOR> -> [ '-' ] <ATOM>
    def _factor(self):
        sign = -1 if self.accept('MINUS') else 1
        return sign * self._atom()

    # <ATOM> -> <BASE> { '^' <EXPONENT> }*
    def _atom(self):
        stack = [self._base()]
        while self.accept('CARET'):
            stack.append(self._exponent())
        if len(stack) == 1: stack.append(1)
        expr = stack.pop()
        for subexpr in reversed(stack):
            exponential = (subexpr == Function('Tensor')(Symbol('e', real=True)))
            expr = exp(expr) if exponential else subexpr ** expr
        return expr

    # <BASE> -> <BASE> -> <NUMBER> | <COMMAND> | <OPERATOR> | <SUBEXPR>
    def _base(self):
        if self.peek('LETTER') or self.peek('SYMB_CMD'):
            self.scanner.mark()
            symbol = self._strip(self._symbol())
            if symbol in ('epsilon', 'Gamma', 'D'):
                self.scanner.reset()
                return self._operator()
            if symbol in self._namespace:
                variable = self._namespace[symbol]
                if isinstance(variable, IndexedSymbol) and variable.rank > 0:
                    self.scanner.reset()
                    return self._operator()
            for key in self._namespace:
                base_symbol = key
                for i, character in enumerate(reversed(base_symbol)):
                    if character not in ('U', 'D'):
                        base_symbol = base_symbol[:len(base_symbol) - i]; break
                if isinstance(self._namespace[key], IndexedSymbol) and symbol == base_symbol \
                        and self._namespace[key].rank > 0:
                    self.scanner.reset()
                    return self._operator()
            if self.peek('CARET'):
                function = Function('Tensor')(Symbol(symbol, real=True))
                if symbol in self._namespace:
                    if isinstance(self._namespace[symbol], Function('Constant')):
                        return self._namespace[symbol]
                else:
                    self._define_tensor(IndexedSymbol(function))
                return function
            self.scanner.reset()
            return self._operator()
        if any(self.peek(token) for token in
                ('RATIONAL', 'DECIMAL', 'INTEGER', 'PI')):
            return self._number()
        if any(self.peek(token) for token in
                ('FUNC_CMD', 'FRAC_CMD', 'SQRT_CMD', 'NLOG_CMD', 'TRIG_CMD', 'COMMAND')):
            return self._command()
        if any(self.peek(token) for token in
                ('PAR_SYM', 'COV_SYM', 'LIE_SYM', 'COMMENT', 'DIACRITIC')):
            return self._operator()
        if any(self.peek(i) for i in ('LPAREN', 'LBRACK', 'BACKSLASH')):
            return self._subexpr()
        sentence, position = self.scanner.sentence, self.scanner.mark()
        raise ParserError('unexpected \'%s\' at position %d' %
            (sentence[position], position), sentence, position)

    # <EXPONENT> -> <BASE> | '{' <EXPRESSION> '}' | '{' '{' <EXPRESSION> '}' '}'
    def _exponent(self):
        if self.accept('LBRACE'):
            if self.accept('LBRACE'):
                base = self._expression()
                self.expect('RBRACE')
            else: base = self._expression()
            self.expect('RBRACE')
            return base
        return self._base()

    # <SUBEXPR> -> '(' <EXPRESSION> ')' | '[' <EXPRESSION> ']' | '\' '{' <EXPRESSION> '\' '}'
    def _subexpr(self):
        if self.accept('LPAREN'):
            expr = self._expression()
            self.expect('RPAREN')
        elif self.accept('LBRACK'):
            expr = self._expression()
            self.expect('RBRACK')
        elif self.accept('BACKSLASH'):
            self.expect('LBRACE')
            expr = self._expression()
            self.expect('BACKSLASH')
            self.expect('RBRACE')
        else:
            sentence, position = self.scanner.sentence, self.scanner.mark()
            raise ParserError('unexpected \'%s\' at position %d' %
                (sentence[position], position), sentence, position)
        return expr

    # <COMMAND> -> <FUNC> | <FRAC> | <SQRT> | <NLOG> | <TRIG>
    def _command(self):
        command = self.scanner.lexeme
        if self.peek('FUNC_CMD'):
            return self._func()
        if self.peek('FRAC_CMD'):
            return self._frac()
        if self.peek('SQRT_CMD'):
            return self._sqrt()
        if self.peek('NLOG_CMD'):
            return self._nlog()
        if self.peek('TRIG_CMD'):
            return self._trig()
        sentence, position = self.scanner.sentence, self.scanner.mark()
        raise ParserError('unsupported command \'%s\' at position %d' %
            (command, position), sentence, position)

    # <FUNC> -> <FUNC_CMD> '{' <EXPRESSION> '}'
    def _func(self):
        func = self._strip(self.scanner.lexeme)
        self.expect('FUNC_CMD')
        self.expect('LBRACE')
        expr = self._expression()
        self.expect('RBRACE')
        if func == 'exp':
            return exp(expr)
        sentence, position = self.scanner.sentence, self.scanner.mark()
        raise ParserError('unsupported function \'%s\' at position %d' %
            (func, position), sentence, position)

    # <FRAC> -> <FRAC_CMD> '{' <EXPRESSION> '}' '{' <EXPRESSION> '}'
    def _frac(self):
        self.expect('FRAC_CMD')
        self.expect('LBRACE')
        numerator = self._expression()
        self.expect('RBRACE')
        self.expect('LBRACE')
        denominator = self._expression()
        self.expect('RBRACE')
        return numerator / denominator

    # <SQRT> -> <SQRT_CMD> [ '[' <INTEGER> ']' ] '{' <EXPRESSION> '}'
    def _sqrt(self):
        self.expect('SQRT_CMD')
        if self.accept('LBRACK'):
            integer = self.scanner.lexeme
            self.expect('INTEGER')
            root = Rational(1, integer)
            self.expect('RBRACK')
        else: root = Rational(1, 2)
        self.expect('LBRACE')
        expr = self._expression()
        self.expect('RBRACE')
        if root == Rational(1, 2):
            return sqrt(expr)
        return Pow(expr, root)

    # <NLOG> -> <NLOG_CMD> [ '_' ( <NUMBER> | '{' <NUMBER> '}' ) ] '{' <EXPRESSION> '}'
    def _nlog(self):
        func = self._strip(self.scanner.lexeme)
        self.expect('NLOG_CMD')
        if func == 'log':
            if self.accept('UNDERSCORE'):
                if self.accept('LBRACE'):
                    base = self._number()
                    self.expect('RBRACE')
                else:
                    base = self._number()
                base = int(base)
            else: base = 10
        self.expect('LBRACE')
        expr = self._expression()
        self.expect('RBRACE')
        if func == 'ln': return log(expr)
        return log(expr, base)

    # <TRIG> -> <TRIG_CMD> [ '^' ( <NUMBER> | '{' <NUMBER> '}' ) ] '{' <EXPRESSION> '}'
    def _trig(self):
        func = self._strip(self.scanner.lexeme)
        self.expect('TRIG_CMD')
        if self.accept('CARET'):
            if self.accept('LBRACE'):
                exponent = self._number()
                self.expect('RBRACE')
            else:
                exponent = self._number()
            exponent = int(exponent)
        else: exponent = 1
        if   func == 'cosh': trig = acosh if exponent == -1 else cosh
        elif func == 'sinh': trig = asinh if exponent == -1 else sinh
        elif func == 'tanh': trig = atanh if exponent == -1 else tanh
        elif func == 'cos':  trig = acos  if exponent == -1 else cos
        elif func == 'sin':  trig = asin  if exponent == -1 else sin
        elif func == 'tan':  trig = atan  if exponent == -1 else tan
        self.expect('LBRACE')
        expr = self._expression()
        self.expect('RBRACE')
        if exponent == -1: return trig(expr)
        return trig(expr) ** exponent

    # <OPERATOR> -> [ '%' <SUFFIX> <VARIABLE> ] ( <PARDRV> | <COVDRV> | <LIEDRV> | <TENSOR> )
    def _operator(self, location='RHS'):
        global_suffix = self._property['suffix']
        if self.accept('COMMENT'):
            self.expect('SUFFIX')
            suffix = self._variable()
            self._property['suffix'] = suffix
        if not global_suffix and location == 'LHS':
            self._property['suffix'] = 'dD'
        operator = self.scanner.lexeme
        if self.peek('PAR_SYM'):
            pardrv = self._pardrv(location)
            self._property['suffix'] = global_suffix
            return pardrv
        if self.peek('COV_SYM') or self.peek('DIACRITIC') or \
                (self.peek('LETTER') and self.scanner.lexeme == 'D'):
            self.scanner.mark()
            if self.accept('DIACRITIC'):
                self.expect('LBRACE')
                if self.peek('COV_SYM') or (self.peek('LETTER') and self.scanner.lexeme == 'D'):
                    self.scanner.reset()
                    covdrv = self._covdrv(location)
                    self._property['suffix'] = global_suffix
                    return covdrv
                self.scanner.reset()
            else:
                covdrv = self._covdrv(location)
                self._property['suffix'] = global_suffix
                return covdrv
        if self.peek('LIE_SYM'):
            liedrv = self._liedrv(location)
            self._property['suffix'] = global_suffix
            return liedrv
        if any(self.peek(token) for token in ('LETTER', 'DIACRITIC', 'SYMB_CMD')):
            tensor = self._tensor(location)
            self._property['suffix'] = global_suffix
            return tensor
        sentence, position = self.scanner.sentence, self.scanner.mark()
        raise ParserError('unsupported operator \'%s\' at position %d' %
            (operator, position), sentence, position)

    # <PARDRV> -> <PAR_SYM> '_' <INDEXING_2> ( <OPERATOR> | <SUBEXPR> )
    def _pardrv(self, location='RHS'):
        self.expect('PAR_SYM')
        self.expect('UNDERSCORE')
        sentence, position = self.scanner.sentence, self.scanner.mark()
        index = self._indexing_2()
        if any(self.peek(i) for i in ('LPAREN', 'LBRACK', 'LBRACE')):
            subexpr = self._subexpr()
            tree = ExprTree(subexpr)
            # insert temporary symbol '_x' for symbolic differentiation
            for subtree in tree.preorder():
                subexpr = subtree.expr
                if subexpr.func in (Function('Tensor'), Derivative):
                    subtree.expr = Function('Function')(subexpr, Symbol('_x'))
                    del subtree.children[:]
            expr = tree.reconstruct()
            # differentiate the expression, including product rule expansion
            tree = ExprTree(diff(expr, Symbol('_x')))
            # remove temporary symbol '_x' from tensor function
            for subtree in tree.preorder():
                subexpr = subtree.expr
                if subexpr.func == Derivative:
                    function = subexpr.args[0].args[0]
                    if function.func == Derivative:
                        subtree.expr = Derivative(function, index)
                    else:
                        symbol = str(function.args[0])
                        tensor = self._namespace[symbol]
                        global_suffix = self._property['suffix']
                        suffix = tensor.suffix
                        if global_suffix and suffix:
                            suffix = global_suffix
                        if not isinstance(index, Symbol) and suffix is None:
                            raise ParserError('cannot perform numeric indexing on a symbolic derivative', sentence, position)
                        subtree.expr = self._define_pardrv(function, location, suffix, index)
                    del subtree.children[:]
                elif subexpr.func == Function('Function'):
                    subtree.expr = subexpr.args[0]
                    del subtree.children[:]
            return tree.reconstruct()
        function = self._operator()
        if function.func == Derivative:
            return Derivative(function, index)
        symbol = str(function.args[0])
        tensor = self._namespace[symbol]
        global_suffix = self._property['suffix']
        suffix = tensor.suffix
        if global_suffix and suffix:
            suffix = global_suffix
        if not isinstance(index, Symbol) and suffix is None:
            raise ParserError('cannot perform numeric indexing on a symbolic derivative', sentence, position)
        return self._define_pardrv(function, location, suffix, index)

    # <COVDRV> -> ( <COV_SYM> | <DIACRITIC> '{' <COV_SYM> '}' ) ( '^' | '_' ) <INDEXING_2> ( <OPERATOR> | <SUBEXPR> )
    def _covdrv(self, location='RHS'):
        diacritic, position = '', self.scanner.mark()
        if self.peek('DIACRITIC'):
            diacritic = self._strip(self.scanner.lexeme)
            self.expect('DIACRITIC')
            operator = '\\' + diacritic + '{\\nabla}'
            self.expect('LBRACE')
            if self.peek('LETTER') and self.scanner.lexeme == 'D':
                self.scanner.lex()
            else: self.expect('COV_SYM')
            self.expect('RBRACE')
        else:
            operator = '\\nabla'
            if self.peek('LETTER') and self.scanner.lexeme == 'D':
                self.scanner.lex()
            else: self.expect('COV_SYM')
        metric = self._property['metric'][diacritic] + diacritic
        if metric + 'DD' not in self._namespace:
            raise ParserError('cannot generate covariant derivative without defined metric \'%s\'' %
                metric, self.scanner.sentence, position)
        if len(metric) > 1:
            metric = '\\mathrm{' + metric + '}'
        if self.accept('CARET'):
            index = (self._indexing_2(), 'U')
        elif self.accept('UNDERSCORE'):
            index = (self._indexing_2(), 'D')
        else:
            sentence, position = self.scanner.sentence, self.scanner.mark()
            raise ParserError('unexpected \'%s\' at position %d' %
                (sentence[position], position), sentence, position)
        func_list, expression = self._expand_product(location, 'cd' + diacritic, index[1], index[0])
        for symbol, function in func_list:
            if index[1] == 'U':
                equation = [operator, ' = ', '', operator]
                idx_gen = IndexedSymbol.index_count()
                indexing = [str(i) for i in function.args[1:]] + [str(index[0])]
                for i, idx in enumerate(indexing):
                    if idx in indexing[:i]:
                        indexing[i] = next(x for x in idx_gen if x not in indexing)
                latex = IndexedSymbol.latex_format(Function('Tensor')(function.args[0],
                            *(Symbol(i) for i in indexing[:-1])))
                covdrv_index = indexing[-1]
                if '_' in str(covdrv_index):
                    base, subscript = str(covdrv_index).split('_')
                    if len(base) > 1:
                        covdrv_index = '\\%s_%s' % (base, subscript)
                elif len(str(covdrv_index)) > 1:
                    covdrv_index = '\\' + str(covdrv_index)
                if index[1] == 'U':
                    equation[0] += '^{' + covdrv_index + '} '
                    bound_index = next(x for x in idx_gen if x not in indexing)
                    equation[2] += '%s^{%s %s} ' % (metric, covdrv_index, bound_index)
                    equation[3] += '_{' + bound_index + '} '
                else:
                    equation[0] += '_{' + covdrv_index + '} '
                    equation[3] += '_{' + covdrv_index + '} '
                equation[0], equation[3] = equation[0] + latex, equation[3] + latex
            if location == 'RHS' and (self._property['suffix'] or symbol not in self._namespace):
                with self.scanner.new_context():
                    if index[1] == 'U':
                        config = ' % assign ' + symbol + ' --suffix dD\n'
                        self.parse_latex(''.join(equation) + config)
                    else:
                        self.parse_latex(Generator.generate_covdrv(function, index[0], symbol, diacritic))
        return expression

    # <LIEDRV> -> <LIE_SYM> '_' <SYMBOL> ( <OPERATOR> | <SUBEXPR> )
    def _liedrv(self, location='RHS'):
        self.expect('LIE_SYM')
        self.expect('UNDERSCORE')
        vector = self._strip(self._symbol())
        func_list, expression = self._expand_product(location, 'ld', vector)
        for symbol, function in func_list:
            if location == 'RHS' and (self._property['suffix'] or symbol not in self._namespace):
                sentence, position = self.scanner.sentence, self.scanner.mark()
                symbol = str(function.args[0])
                tensor = IndexedSymbol(function, self._namespace[symbol].dimension)
                tensor.weight = self._namespace[symbol].weight
                self.parse_latex(Generator.generate_liedrv(function, vector, tensor.weight))
                self.scanner.initialize(sentence, position)
                self.scanner.lex()
        return expression

    # <TENSOR> -> <SYMBOL> [ ( '_' <INDEXING_4> ) | ( '^' <INDEXING_3> [ '_' <INDEXING_4> ] ) ]
    def _tensor(self, location='RHS'):
        sentence, position = self.scanner.sentence, self.scanner.mark()
        indexing = []
        symbol = list(self._strip(self._symbol()))
        if self.accept('UNDERSCORE'):
            index, order = self._indexing_4()
            covariant = ';' in index
            for i in (',', ';'):
                if i in index: index.remove(i)
            indexing.extend(index)
            symbol.extend((len(index) - order) * ['D'])
            if order > 0:
                sentence = self.scanner.sentence
                suffix = '_cd' if covariant else '_d'
                symbol.append(suffix + order * 'D')
                function = Function('Tensor')(Symbol(''.join(symbol)), *indexing)
                old_latex = sentence[position:self.scanner.mark()]
                new_latex = IndexedSymbol(function).latex_format(function)
                self.scanner.sentence = sentence.replace(old_latex, new_latex)
                self.scanner.marker = position
                self.scanner.reset()
                return self._operator()
        self.scanner.mark()
        if self.accept('CARET'):
            if self.accept('LBRACE'):
                if self.accept('LBRACE'):
                    self.scanner.reset()
                    symbol = ''.join(symbol)
                    function = Function('Tensor')(Symbol(symbol, real=True))
                    if symbol in self._namespace:
                        if isinstance(self._namespace[symbol], Function('Constant')):
                            return self._namespace[symbol]
                    else: self._define_tensor(IndexedSymbol(function))
                    return function
                self.scanner.reset(); self.scanner.lex()
            index = self._indexing_3()
            indexing.extend(index)
            symbol.extend(len(index) * ['U'])
            if self.accept('UNDERSCORE'):
                index, order = self._indexing_4()
                covariant = ';' in index
                for i in (',', ';'):
                    if i in index: index.remove(i)
                indexing.extend(index)
                symbol.extend((len(index) - order) * ['D'])
                if order > 0:
                    sentence = self.scanner.sentence
                    suffix = '_cd' if covariant else '_d'
                    symbol.append(suffix + order * 'D')
                    function = Function('Tensor')(Symbol(''.join(symbol)), *indexing)
                    old_latex = sentence[position:self.scanner.mark()]
                    new_latex = IndexedSymbol(function).latex_format(function)
                    self.scanner.sentence = sentence.replace(old_latex, new_latex)
                    self.scanner.marker = position
                    self.scanner.reset()
                    return self._operator()
        symbol = ''.join(symbol)
        if symbol in self._namespace:
            if isinstance(self._namespace[symbol], Function('Constant')):
                return self._namespace[symbol]
        function = Function('Tensor')(Symbol(symbol, real=True), *indexing)
        tensor = IndexedSymbol(function)
        if symbol not in self._namespace and location == 'RHS':
            if symbol[:7] == 'epsilon':
                # instantiate permutation (Levi-Civita) symbol using parity
                def sgn(sequence):
                    """ Permutation Signature (Parity)"""
                    cycle_length = 0
                    for n, i in enumerate(sequence[:-1]):
                        for j in sequence[(n + 1):]:
                            if i == j: return 0
                            cycle_length += i > j
                    return (-1)**cycle_length
                index = [chr(105 + n) for n in range(tensor.rank)]
                prefix = '[' * tensor.rank + 'sgn([' + ', '.join(index) + '])'
                suffix = ''.join(' for %s in range(%d)]' % (index[tensor.rank - i], tensor.rank)
                    for i in range(1, tensor.rank + 1))
                tensor.structure = eval(prefix + suffix, {'sgn': sgn})
                tensor.dimension = tensor.rank
                self._define_tensor(tensor)
            else:
                if tensor.rank > 0:
                    if any(suffix in symbol for suffix in ('_d', '_dup', '_cd', '_ld')):
                        raise ParserError('cannot index undefined variable \'%s\' at position %d' %
                            (symbol, position), sentence, position)
                    i, base_symbol = len(symbol) - 1, symbol
                    while i >= 0:
                        if base_symbol[i] not in ('U', 'D'):
                            base_symbol = base_symbol[:(i + 1)]
                            break
                        i -= 1
                    for suffix in product(*('UD' if i == 'U' else 'DU' for _, i in IndexedSymbol.indexing(function))):
                        symbol_RHS = base_symbol + ''.join(suffix)
                        if symbol_RHS in self._namespace:
                            with self.scanner.new_context():
                                diacritic = 'bar'   if 'bar'   in symbol \
                                       else 'hat'   if 'hat'   in symbol \
                                       else 'tilde' if 'tilde' in symbol \
                                       else ''
                                metric = self._namespace[symbol_RHS].metric if self._namespace[symbol_RHS].metric else \
                                    self._property['metric'][diacritic] + diacritic
                                if metric + 'DD' not in self._namespace:
                                    raise ParserError('cannot raise/lower index for \'%s\' without defined metric at position %d' %
                                        (symbol, position), sentence, position)
                                indexing_LHS = indexing_RHS = [str(index) for index in indexing]
                                idx_gen = IndexedSymbol.index_count()
                                for i, index in enumerate(indexing_LHS):
                                    if index in indexing_LHS[:i]:
                                        indexing_LHS[i] = next(x for x in idx_gen if x not in indexing_LHS)
                                function_LHS = Function('Tensor')(function.args[0],
                                    *(Symbol(i) for i in indexing_LHS))
                                latex = IndexedSymbol.latex_format(function_LHS) + ' = '
                                for i, (idx, pos) in enumerate(IndexedSymbol.indexing(function_LHS)):
                                    if pos != suffix[i]:
                                        indexing_RHS[i] = next(x for x in idx_gen if x not in indexing_LHS)
                                        if '_' in str(idx):
                                            base, subscript = str(idx).split('_')
                                            if len(base) > 1:
                                                idx = '\\%s_%s' % (base, subscript)
                                        elif len(str(idx)) > 1:
                                            idx = '\\' + str(idx)
                                        if pos == 'U':
                                            latex += '\\mathrm{%s}^{%s %s} ' % (metric, idx, indexing_RHS[i])
                                        else:
                                            latex += '\\mathrm{%s}_{%s %s} ' % (metric, idx, indexing_RHS[i])
                                latex += IndexedSymbol.latex_format(Function('Tensor')(Symbol(symbol_RHS, real=True), *indexing_RHS))
                                suffix = self._namespace[symbol_RHS].suffix
                                if suffix or self._namespace[symbol_RHS].metric:
                                    latex += ' % assign ' + symbol
                                    if suffix:
                                        latex += ' --suffix ' + suffix + ' '
                                    if self._namespace[symbol_RHS].metric:
                                        latex += ' --metric ' + metric + ' '
                                self.parse_latex(latex)
                            return function
                    raise ParserError('cannot index undefined variable \'%s\' at position %d' %
                        (symbol, position), sentence, position)
                else: self._define_tensor(tensor)
        return function

    # <SYMBOL> -> <LETTER> | <SYMB_CMD> '{' <VARIABLE> '}' | <DIACRITIC> '{' <SYMBOL> '}'
    def _symbol(self):
        lexeme = self.scanner.lexeme
        if self.accept('LETTER'):
            return lexeme
        if self.accept('SYMB_CMD'):
            self.expect('LBRACE')
            symbol = self._variable()
            self.expect('RBRACE')
            return symbol
        if self.accept('DIACRITIC'):
            self.expect('LBRACE')
            symbol = self._symbol() + lexeme[1:]
            self.expect('RBRACE')
            return symbol
        sentence, position = self.scanner.sentence, self.scanner.mark()
        raise ParserError('unexpected \'%s\' at position %d' %
            (sentence[position], position), sentence, position)

    # <INDEXING_1> -> <LETTER> [ '_' <INDEXING_2> ] | <INTEGER>
    def _indexing_1(self):
        lexeme = self._strip(self.scanner.lexeme)
        if self.accept('LETTER'):
            index = Symbol(lexeme, real=True) if not self.accept('UNDERSCORE') \
                else Symbol('%s_%s' % (lexeme, self._indexing_2()), real=True)
            return index if index not in self._property['coord'] \
                else Integer(self._property['coord'].index(index))
        elif self.accept('INTEGER'):
            return Integer(lexeme)
        sentence, position = self.scanner.sentence, self.scanner.mark()
        raise ParserError('unexpected \'%s\' at position %d' %
            (sentence[position], position), sentence, position)

    # <INDEXING_2> -> <LETTER> | <INTEGER> | '{' <INDEXING_1> '}'
    def _indexing_2(self):
        lexeme = self._strip(self.scanner.lexeme)
        if self.accept('LETTER'):
            index = Symbol(lexeme, real=True)
            return index if index not in self._property['coord'] \
                else Integer(self._property['coord'].index(index))
        elif self.accept('INTEGER'):
            return Integer(lexeme)
        elif self.accept('LBRACE'):
            indexing = self._indexing_1()
            self.expect('RBRACE')
            return indexing
        sentence, position = self.scanner.sentence, self.scanner.mark()
        raise ParserError('unexpected \'%s\' at position %d' %
            (sentence[position], position), sentence, position)

    # <INDEXING_3> -> <INDEXING_2> | '{' { <INDEXING_1> }+ '}'
    def _indexing_3(self):
        indexing = []
        if self.accept('LBRACE'):
            while not self.accept('RBRACE'):
                indexing.append(self._indexing_1())
            return indexing
        return [self._indexing_2()]

    # <INDEXING_4> -> <INDEXING_2> | '{' ( ',' | ';' ) { <INDEXING_1> }+ | { <INDEXING_1> }+ [ ( ',' | ';' ) { <INDEXING_1> }+ ] '}'
    def _indexing_4(self):
        indexing, order = [], 0
        if self.accept('LBRACE'):
            lexeme = self.scanner.lexeme
            if self.accept('COMMA') or self.accept('SEMICOLON'):
                indexing.append(lexeme)
                while not self.accept('RBRACE'):
                    indexing.append(self._indexing_1())
                    order += 1
                return indexing, order
            while not any(self.peek(i) for i in ('RBRACE', 'COMMA', 'SEMICOLON')):
                indexing.append(self._indexing_1())
            lexeme = self.scanner.lexeme
            if self.accept('COMMA') or self.accept('SEMICOLON'):
                indexing.append(lexeme)
                while not self.accept('RBRACE'):
                    indexing.append(self._indexing_1())
                    order += 1
            else: self.expect('RBRACE')
            return indexing, order
        return [self._indexing_2()], order

    # <VARIABLE> -> <LETTER> { [ '_' ] ( <LETTER> | <INTEGER> ) }*
    def _variable(self):
        variable = self.scanner.lexeme
        self.expect('LETTER')
        while any(self.peek(token) for token in ('UNDERSCORE', 'LETTER', 'INTEGER')):
            variable += self.scanner.lexeme
            self.scanner.lex()
        return variable

    # <NUMBER> -> [ '-' ] ( <RATIONAL> | <DECIMAL> | <INTEGER> | <PI> )
    def _number(self):
        sign = -1 if self.accept('MINUS') else 1
        number = self.scanner.lexeme
        if self.accept('RATIONAL'):
            rational = re.match(r'([0-9]+)\/([1-9][0-9]*)', number)
            return sign * Rational(rational.group(1), rational.group(2))
        if self.accept('DECIMAL'):
            return sign * Float(number)
        if self.accept('INTEGER'):
            return sign * Integer(number)
        if self.accept('PI'):
            return sign * pi
        sentence, position = self.scanner.sentence, self.scanner.mark()
        raise ParserError('unexpected \'%s\' at position %d' %
            (sentence[position], position), sentence, position)

    def _define_tensor(self, tensor, zeros=False):
        symbol, dimension = tensor.symbol, tensor.dimension
        symmetry = tensor.symmetry if tensor.rank > 1 else None
        if not tensor.structure:
            tensor.structure = Symbol(symbol, real=True) if tensor.rank == 0 \
                else symdef(tensor.rank, symbol if not zeros else None, symmetry, dimension)
        self._namespace[symbol] = tensor
        self.state[symbol] = None

    def _define_pardrv(self, function, location, suffix, index):
        if suffix is None:
            return Derivative(function, index)
        symbol, indices = str(function.args[0]), list(function.args[1:]) + [index]
        suffix = '_' + suffix[:-1]
        tensor, index = self._namespace[symbol], str(index)
        symbol = symbol + ('' if suffix in symbol else suffix) + 'D'
        with self.scanner.new_context():
            if symbol not in self._namespace:
                if location == 'RHS' and tensor.equation:
                    LHS, RHS = tensor.equation[0]
                    tree, idx_set = ExprTree(tensor.equation[1]), set()
                    for subtree in tree.preorder():
                        subexpr = subtree.expr
                        if subexpr.func == Function('Tensor'):
                            idx_set.update(subexpr.args[1:])
                    idx_set = {str(i) for i in idx_set}
                    if index in idx_set:
                        idx_gen = IndexedSymbol.index_count()
                        index = next(x for x in idx_gen if x not in idx_set)
                    if '_' in str(index):
                        base, subscript = str(index).split('_')
                        if len(base) > 1:
                            index = '\\%s_%s' % (base, subscript)
                    elif len(str(index)) > 1:
                        index = '\\' + str(index)
                    impsum = '' if tensor.impsum else ' % noimpsum'
                    self.parse_latex('\\partial_{%s} %s = \\partial_{%s} (%s)%s'
                        % (index, LHS.strip(), index, RHS.rstrip(' \n\\'), impsum))
        function = Function('Tensor')(Symbol(symbol, real=True), *indices)
        if symbol not in self._namespace:
            symmetry = 'nosym'
            if len(symbol.split(suffix)[1]) == 2:
                position = len(indices) - 2
                symmetry = 'sym%d%d' % (position, position + 1)
            if tensor.symmetry and tensor.symmetry != 'nosym':
                symmetry = tensor.symmetry + ('_' + symmetry if symmetry != 'nosym' else '')
            self._define_tensor(IndexedSymbol(function, tensor.dimension,
                symmetry=symmetry, suffix=tensor.suffix))
        return function

    def _expand_product(self, location, suffix_1, suffix_2, index=None):
        func_list, product = [], None
        if any(self.peek(i) for i in ('LPAREN', 'LBRACK', 'LBRACE')):
            subexpr = self._subexpr()
            tree = ExprTree(subexpr)
            # insert temporary symbol '_x' for symbolic differentiation
            for subtree in tree.preorder():
                subexpr = subtree.expr
                if subexpr.func in (Function('Tensor'), Derivative):
                    subtree.expr = Function('Function')(subexpr, Symbol('_x'))
                    del subtree.children[:]
            expr = tree.reconstruct()
            # differentiate the expression, including product rule expansion
            tree = ExprTree(diff(expr, Symbol('_x')))
            # remove temporary symbol '_x' from tensor function
            for subtree in tree.preorder():
                subexpr = subtree.expr
                if subexpr.func == Derivative:
                    function = subexpr.args[0].args[0]
                    if function.func == Derivative:
                        base_func = function.args[0]
                    else: base_func = function
                    symbol, indices = str(base_func.args[0]), list(base_func.args[1:])
                    if function.func == Derivative:
                        symbol += '_dD'
                        for pardrv_index, _ in function.args[1:]:
                            indices.append(pardrv_index)
                        function = Function('Tensor')(Symbol(symbol, real=True), *indices)
                    if index: indices.append(index)
                    name_list = re.split(r'_(cd|ld)', symbol)
                    if len(name_list) > 1:
                        if name_list[-2] == 'cd' != suffix_1[:2] or \
                           name_list[-2] == 'ld' != suffix_1[:2]:
                            symbol += '_' + suffix_1
                    else: symbol += '_' + suffix_1
                    symbol += suffix_2
                    subtree.expr = Function('Tensor')(Symbol(symbol, real=True), *indices)
                    func_list.append((symbol, function))
                    del subtree.children[:]
                elif subexpr.func == Function('Function'):
                    subtree.expr = subexpr.args[0]
                    del subtree.children[:]
            product = tree.reconstruct()
        else:
            function = self._operator(location)
            if function.func == Derivative:
                base_func = function.args[0]
            else: base_func = function
            symbol, indices = str(base_func.args[0]), list(base_func.args[1:])
            if function.func == Derivative:
                symbol += '_dD'
                for pardrv_index, _ in function.args[1:]:
                    indices.append(pardrv_index)
                function = Function('Tensor')(Symbol(symbol, real=True), *indices)
            if index: indices.append(index)
            name_list = re.split(r'_(cd|ld)', symbol)
            if len(name_list) > 1:
                if name_list[-2] == 'cd' != suffix_1[:2] or \
                   name_list[-2] == 'ld' != suffix_1[:2]:
                    symbol += '_' + suffix_1
            else: symbol += '_' + suffix_1
            symbol += suffix_2
            func_list.append((symbol, function))
            product = Function('Tensor')(Symbol(symbol, real=True), *indices)
        return func_list, product

    @staticmethod
    def _strip(symbol):
        return symbol[1:] if symbol[0] == '\\' else symbol

    def peek(self, token):
        if self.scanner.token is None and token == 'LINEBREAK':
            return True
        return self.scanner.token == token

    def accept(self, token):
        if self.peek(token):
            self.scanner.lex()
            return True
        return False

    def expect(self, token):
        if not self.accept(token):
            sentence, position = self.scanner.sentence, self.scanner.mark()
            raise ParserError('expected token %s at position %d' %
                (token, position), sentence, position)

class ParserError(NRPyLaTeXError):

    def __init__(self, message, sentence=None, position=None):
        super(ParserError, self).__init__(message, sentence, position)
