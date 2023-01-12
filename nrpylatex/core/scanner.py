""" NRPyLaTeX Scanner """
# Author: Ken Sible
# Email:  ksible *at* outlook *dot* com

import re

class Scanner:
    """ The following class will tokenize a LaTeX sentence for parsing. """

    def __init__(self):
        # define a regex pattern for every token, create a named capture group for
        # every pattern, join together the resulting pattern list using a pipe symbol
        # for regex alternation, and compile the generated regular expression
        symmetry = r'nosym|(?:sym|anti)[0-9]+(?:_(?:sym|anti)[0-9]+)*'
        alphabet = '|'.join(letter for letter in (r'\\[aA]lpha', r'\\[bB]eta', r'\\[gG]amma', r'\\[dD]elta',
            r'\\[eE]psilon', r'\\[zZ]eta', r'\\[eE]ta', r'\\[tT]heta', r'\\[iI]ota', r'\\[kK]appa', r'\\[lL]ambda',
            r'\\[mM]u', r'\\[nN]u', r'\\[xX]i', r'\\[oO]mikron', r'\\[pP]i', r'\\[Rr]ho', r'\\[sS]igma', r'\\[tT]au',
            r'\\[uU]psilon', r'\\[pP]hi', r'\\[cC]hi', r'\\[pP]si', r'\\[oO]mega', r'\\varepsilon', r'\\varkappa',
            r'\\varphi', r'\\varpi', r'\\varrho', r'\\varsigma', r'\\vartheta', r'[a-zA-Z]'))
        self.token_dict = [
            ('LINEBREAK',       r'\r?\n'),
            ('WHITESPACE',      r'\s+'),
            ('SYMMETRY',        symmetry),
            ('STRING',          r'\"[^\"]*\"'),
            ('GROUP',           r'\<[0-9]+(\.{2})?\>'),
            ('RATIONAL',        r'\-?[0-9]+\/\-?[1-9][0-9]*'),
            ('DECIMAL',         r'\-?[0-9]+\.[0-9]+'),
            ('INTEGER',         r'\-?[0-9]+'),
            ('ARROW',           r'\-\>'),
            ('PLUS',            r'\+'),
            ('MINUS',           r'\-'),
            ('DIVIDE',          r'\/'),
            ('EQUAL',           r'\='),
            ('CARET',           r'\^'),
            ('UNDERSCORE',      r'\_'),
            ('COMMENT',         r'\%'),
            ('PRIME',           r'\''),
            ('COMMA',           r'\,'),
            ('COLON',           r'\:'),
            ('SEMICOLON',       r'\;'),
            ('LPAREN',          r'\('),
            ('RPAREN',          r'\)'),
            ('LBRACK',          r'\['),
            ('RBRACK',          r'\]'),
            ('LBRACE',          r'\{'),
            ('RBRACE',          r'\}'),
            ('PAR_SYM',         r'\\partial'),
            ('COV_SYM',         r'\\nabla'),
            ('LIE_SYM',         r'\\mathcal\{L\}'),
            ('SYMB_CMD',        r'\\mathrm|\\text'),
            ('FUNC_CMD',        r'\\exp'),
            ('FRAC_CMD',        r'\\frac'),
            ('SQRT_CMD',        r'\\sqrt'),
            ('NLOG_CMD',        r'\\ln|\\log'),
            ('TRIG_CMD',        r'\\sinh|\\cosh|\\tanh|\\sin|\\cos|\\tan'),
            ('DEFINE_MACRO',    r'define'),
            ('ASSIGN_MACRO',    r'assign'),
            ('IGNORE_MACRO',    r'ignore'),
            ('SREPL_MACRO',     r'srepl'),
            ('INDEX_MACRO',     r'index'),
            ('COORD_MACRO',     r'coord'),
            ('ZERO',            r'zero'),
            ('KRON',            r'kron'),
            ('CONST',           r'const'),
            ('DIM',             r'dim'),
            ('SYM',             r'sym'),
            ('WEIGHT',          r'weight'),
            ('DERIV',           r'deriv'),
            ('METRIC',          r'metric'),
            ('DEFAULT',         r'default'),
            ('PERSIST',         r'persist'),
            ('NOIMPSUM',        r'noimpsum'),
            ('SUFFIX',          r'none|dD|dupD'),
            ('DIACRITIC',       r'\\hat|\\tilde|\\bar'),
            ('PI',              r'\\pi'),
            ('LETTER',          alphabet),
            ('COMMAND',         r'\\[a-zA-Z]+'),
            ('NEWLINE',         r'\\{2}'),
            ('BACKSLASH',       r'\\')]
        self.regex = re.compile('|'.join(['(?P<%s>%s)' % pattern for pattern in self.token_dict]))
        self.token_dict = dict(self.token_dict)

    def initialize(self, sentence, position=0, whitespace=False):
        """ Initialize Scanner

            :arg: sentence (raw string)
            :arg: position
        """
        self.sentence = sentence
        self.token    = None
        self.lexeme   = None
        self.marker   = None
        self.index    = position
        self._whitespace = whitespace

    def tokenize(self):
        """ Tokenize Sentence

            :return: token iterator
        """
        while self.index < len(self.sentence):
            token = self.regex.match(self.sentence, self.index)
            if token is None:
                raise ScanError('unexpected \'%s\' at position %d' %
                    (self.sentence[self.index], self.index), self.sentence, self.index)
            self.index = token.end()
            if self.whitespace or token.lastgroup not in ('WHITESPACE', 'LINEBREAK'):
                self.lexeme = token.group()
                yield token.lastgroup

    def lex(self):
        """ Retrieve Next Token

            :return: next token
        """
        try:
            self.token = next(self.tokenize())
        except StopIteration:
            self.token  = None
            self.lexeme = ''
        return self.token

    def mark(self):
        """ Mark Iterator Position

            :return: previous position
        """
        self.marker = self.index - len(self.lexeme)
        return self.marker

    def reset(self, index=None):
        """ Reset Token Iterator """
        if not self.sentence:
            raise RuntimeError('cannot reset uninitialized scanner')
        self.initialize(self.sentence, self.marker if index is None else index, self.whitespace)
        self.lex()

    @property
    def whitespace(self):
        return self._whitespace

    @whitespace.setter
    def whitespace(self, flag):
        if not flag:
            while self.token in ('WHITESPACE', 'LINEBREAK'):
                self.lex()
        self._whitespace = flag

    def new_context(self):
        return self.ScannerContext(self)

    class ScannerContext():

        def __init__(self, scanner):
            self.scanner = scanner
            self.state = (scanner.sentence, scanner.mark(), scanner.whitespace)

        def __enter__(self): return

        def __exit__(self, exc_type, exc_value, exc_tb):
            self.scanner.initialize(*self.state)
            self.scanner.lex()

class ScanError(Exception):
    """ Invalid LaTeX Sentence """

    def __init__(self, message, sentence=None, position=None):
        if position is not None:
            length = 0
            for _, substring in enumerate(sentence.split('\n')):
                if position - length <= len(substring):
                    sentence = substring.lstrip()
                    position += len(sentence) - len(substring) - length
                    break
                length += len(substring) + 1
            super(ScanError, self).__init__('%s\n%s^\n' % (sentence, (12 + position) * ' ') + message)
        else: super(ScanError, self).__init__(message)
