""" Expression Trees, Coordinate Systems and Indexed Symbols """
# Author: Ken Sible
# Email:  ksible *at* outlook *dot* com

from nrpylatex.utils.functional import product
from sympy import Function, Symbol, sympify
import sys, re

class ExprTree:
    """ Symbolic Expression Tree

        >>> from sympy.abc import a, b, x
        >>> from sympy import cos
        >>> tree = ExprTree(cos(a + b)**2)
        >>> print(tree)
        ExprTree(cos(a + b)**2)
        >>> [node.expr for node in tree.preorder()]
        [cos(a + b)**2, cos(a + b), a + b, a, b, 2]
    """

    def __init__(self, expr):
        self.root = self.Node(expr, None)
        self.build(self.root)

    def build(self, node, clear=True):
        """ Build expression (sub)tree.

            :arg:   root node of (sub)tree
            :arg:   clear children (default: True)

            >>> from sympy.abc import a, b
            >>> from sympy import cos, sin
            >>> tree = ExprTree(cos(a + b)**2)
            >>> tree.root.expr = sin(a*b)**2
            >>> tree.build(tree.root, clear=True)
            >>> [node.expr for node in tree.preorder()]
            [sin(a*b)**2, sin(a*b), a*b, a, b, 2]
        """
        if clear: del node.children[:]
        for arg in node.expr.args:
            subtree = self.Node(arg, node.expr.func)
            node.append(subtree)
            self.build(subtree)

    def preorder(self, node=None):
        """ Generate iterator for preorder traversal.

            :arg:    root node of (sub)tree
            :return: iterator

            >>> from sympy.abc import a, b
            >>> from sympy import cos, Mul
            >>> tree = ExprTree(cos(a*b)**2)
            >>> for i, subtree in enumerate(tree.preorder()):
            ...     if subtree.expr.func == Mul:
            ...         print((i, subtree.expr))
            (2, a*b)
        """
        if node is None:
            node = self.root
        yield node
        for child in node.children:
            for subtree in self.preorder(child):
                yield subtree

    def postorder(self, node=None):
        """ Generate iterator for postorder traversal.

            :arg:    root node of (sub)tree
            :return: iterator

            >>> from sympy.abc import a, b
            >>> from sympy import cos, Mul
            >>> tree = ExprTree(cos(a*b)**2)
            >>> for i, subtree in enumerate(tree.postorder()):
            ...     if subtree.expr.func == Mul:
            ...         print((i, subtree.expr))
            (2, a*b)
        """
        if node is None:
            node = self.root
        for child in node.children:
            for subtree in self.postorder(child):
                yield subtree
        yield node

    def reconstruct(self, evaluate=False):
        """
        Reconstruct root expression from expression tree.

        :arg:    evaluate root expression (default: False)
        :return: root expression

        >>> from sympy.abc import a, b
        >>> from sympy import cos, sin
        >>> tree = ExprTree(cos(a + b)**2)
        >>> tree.root.children[0].expr = sin(a + b)
        >>> tree.reconstruct()
        sin(a + b)**2
        """
        for subtree in self.postorder():
            if subtree.children:
                expr_list = [node.expr for node in subtree.children]
                subtree.expr = subtree.expr.func(*expr_list, evaluate=evaluate)
        return self.root.expr

    class Node:
        """ Expression Tree Node """
        def __init__(self, expr, func):
            self.expr = expr
            self.func = func
            self.children = []

        def append(self, node):
            self.children.append(node)

        def __repr__(self):
            return 'Node(%s, %s)' % (self.expr, self.func)

        def __str__(self):
            return str(self.expr)

    def __repr__(self):
        return 'ExprTree(' + str(self.root.expr) + ')'

    __str__ = __repr__

class CoordinateSystem(list):

    def __init__(self, symbol):
        self.symbol = symbol

    def default(self, n):
        return Symbol('%s_%d' % (self.symbol, n), real=True)

    def index(self, value):
        pattern = re.match(self.symbol + '_([0-9][0-9]*)', str(value))
        if pattern is not None:
            return pattern.group(1)
        return list.index(self, value)

    def __missing__(self, index):
        return self.default(index)

    def __getitem__(self, index):
        try:
            return list.__getitem__(self, index)
        except IndexError:
            return self.__missing__(index)

    def __contains__(self, key):
        return list.__contains__(self, key) or re.match(self.symbol + '_[0-9][0-9]*', str(key))

    def __eq__(self, other):
        return list.__eq__(self, other) and self.symbol == other.symbol

class IndexedSymbol:

    def __init__(self, function, dimension=None, structure=None, equation=None,
            symmetry=None, suffix=None, metric=None, weight=None, impsum=True):
        self.overridden  = False
        self.symbol      = str(function.args[0])
        self.rank        = 0
        for symbol in re.split(r'_d[^UD]*|_cd|_ld', self.symbol):
            for character in reversed(symbol):
                if character in ('U', 'D'):
                    self.rank += 1
                else: break
        self.dimension   = dimension
        self.structure   = structure
        self.equation    = equation
        self.symmetry    = symmetry
        self.suffix      = suffix
        self.metric      = metric
        self.weight      = weight
        self.impsum      = impsum

    @staticmethod
    def indexing(function):
        """ Symbol Indexing from SymPy Function """
        symbol, indices = function.args[0], function.args[1:]
        i, indexing = len(indices) - 1, []
        for symbol in reversed(re.split(r'_d[^UD]*|_cd|_ld', str(symbol))):
            for character in reversed(symbol):
                if character in ('U', 'D'):
                    indexing.append((indices[i], character))
                else: break
                i -= 1
        return list(reversed(indexing))

    # TODO change method type to static (class) method
    def array_format(self, function):
        """ Indexed Symbol Notation for Array Formatting """
        if isinstance(function, Function('Tensor')):
            indexing = self.indexing(function)
        else: indexing = function
        if not indexing:
            return self.symbol
        return self.symbol + ''.join(['[' + str(index) + ']' for index, _ in indexing])

    @staticmethod
    def latex_format(function):
        """ Indexed Symbol Notation for LaTeX Formatting """
        symbol, indexing = str(function.args[0]), IndexedSymbol.indexing(function)
        operator, i_2 = '', len(symbol)
        for i_1 in range(len(symbol), 0, -1):
            subsym = symbol[i_1:i_2]
            if '_d' in subsym:
                suffix = re.split(r'_d[^UD]*', subsym)[-1]
                for _ in reversed(suffix):
                    index = str(indexing.pop()[0])
                    if '_' in index:
                        base, subscript = index.split('_')
                        if len(base) > 1:
                            index = '\\%s_{%s}' % (base, subscript)
                    elif len(index) > 1:
                        index = '\\' + index
                    operator += '\\partial_{' + index + '} '
                i_2 = i_1
            elif '_cd' in subsym:
                suffix = subsym.split('_cd')[-1]
                diacritic = 'bar'   if 'bar'   in suffix \
                       else 'hat'   if 'hat'   in suffix \
                       else 'tilde' if 'tilde' in suffix \
                       else None
                if diacritic:
                    suffix = suffix[len(diacritic):]
                for position in reversed(suffix):
                    index = str(indexing.pop()[0])
                    if '_' in index:
                        base, subscript = index.split('_')
                        if len(base) > 1:
                            index = '\\%s_{%s}' % (base, subscript)
                    elif len(index) > 1:
                        index = '\\' + index
                    operator += '\\' + diacritic + '{\\nabla}' if diacritic \
                        else '\\nabla'
                    if position == 'U':
                        operator += '^{' + index + '}'
                    else:
                        operator += '_{' + index + '}'
                    operator += ' '
                i_2 = i_1
            elif '_ld' in subsym:
                vector = re.split('_ld', subsym)[-1]
                if len(vector) > 1:
                    vector = '\\mathrm{' + vector + '}'
                operator += '\\mathcal{L}_' + vector + ' '
                i_2 = i_1
        symbol = re.split(r'_d[^UD]*|_cd|_ld', symbol)[0]
        for i, character in enumerate(reversed(symbol)):
            if character not in ('U', 'D'):
                symbol = symbol[:len(symbol) - i]; break
        latex = [symbol, [], []]
        if len(latex[0]) > 1:
            latex[0] = '\\mathrm{' + str(latex[0]) + '}'
        latex[0] = operator + latex[0]
        U_count, D_count = 0, 0
        for index, position in indexing:
            index = str(index)
            if '_' in index:
                base, subscript = index.split('_')
                if len(base) > 1:
                    index = '\\%s_{%s}' % (base, subscript)
            elif len(index) > 1:
                index = '\\' + index
            if position == 'U':
                latex[1].append(index)
                U_count += 1
            else:
                latex[2].append(index)
                D_count += 1
        latex[1] = ' '.join(latex[1])
        latex[2] = ' '.join(latex[2])
        if U_count > 0:
            latex[1] = '^{' + latex[1] + '}'
        if D_count > 0:
            latex[2] = '_{' + latex[2] + '}'
        return ''.join(latex)

    @staticmethod
    def index_count():
        n = 1
        while True:
            yield 'i_' + str(n)
            n += 1

    def __repr__(self):
        symbol = ('*' if self.overridden else '') + self.symbol
        if self.rank == 0:
            return 'Scalar(%s)' % symbol
        return 'Tensor(%s, %dD)' % (symbol, self.dimension)

    __str__ = __repr__

class IndexedSymbolError(Exception):
    """ Invalid Indexed Symbol """

def symdef(rank, symbol=None, symmetry=None, dimension=None):
    """ Generate an indexed symbol of specified rank and dimension

        >>> indexed_symbol =  symdef(rank=2, symbol='M', dimension=3, symmetry='sym01')
        >>> assert pipe(indexed_symbol, lambda x: repeat(flatten, x, 1), set, len) == 6

        >>> indexed_symbol =  symdef(rank=3, symbol='M', dimension=3, symmetry='sym01')
        >>> assert len(set(repeat(flatten, indexed_symbol, 2))) == 18
        >>> indexed_symbol =  symdef(rank=3, symbol='M', dimension=3, symmetry='sym02')
        >>> assert len(set(repeat(flatten, indexed_symbol, 2))) == 18
        >>> indexed_symbol =  symdef(rank=3, symbol='M', dimension=3, symmetry='sym12')
        >>> assert len(set(repeat(flatten, indexed_symbol, 2))) == 18

        >>> indexed_symbol =  symdef(rank=3, symbol='M', dimension=3, symmetry='sym012')
        >>> assert len(set(repeat(flatten, indexed_symbol, 2))) == 10

        >>> indexed_symbol =  symdef(rank=4, symbol='M', dimension=3, symmetry='sym01')
        >>> assert len(set(repeat(flatten, indexed_symbol, 3))) == 54
        >>> indexed_symbol =  symdef(rank=4, symbol='M', dimension=3, symmetry='sym02')
        >>> assert len(set(repeat(flatten, indexed_symbol, 3))) == 54
        >>> indexed_symbol =  symdef(rank=4, symbol='M', dimension=3, symmetry='sym03')
        >>> assert len(set(repeat(flatten, indexed_symbol, 3))) == 54
        >>> indexed_symbol =  symdef(rank=4, symbol='M', dimension=3, symmetry='sym12')
        >>> assert len(set(repeat(flatten, indexed_symbol, 3))) == 54
        >>> indexed_symbol =  symdef(rank=4, symbol='M', dimension=3, symmetry='sym13')
        >>> assert len(set(repeat(flatten, indexed_symbol, 3))) == 54
        >>> indexed_symbol =  symdef(rank=4, symbol='M', dimension=3, symmetry='sym23')
        >>> assert len(set(repeat(flatten, indexed_symbol, 3))) == 54

        >>> indexed_symbol =  symdef(rank=4, symbol='M', dimension=3, symmetry='sym012')
        >>> assert len(set(repeat(flatten, indexed_symbol, 3))) == 30
        >>> indexed_symbol =  symdef(rank=4, symbol='M', dimension=3, symmetry='sym013')
        >>> assert len(set(repeat(flatten, indexed_symbol, 3))) == 30
        >>> indexed_symbol =  symdef(rank=4, symbol='M', dimension=3, symmetry='sym01_sym23')
        >>> assert len(set(repeat(flatten, indexed_symbol, 3))) == 36
        >>> indexed_symbol =  symdef(rank=4, symbol='M', dimension=3, symmetry='sym02_sym13')
        >>> assert len(set(repeat(flatten, indexed_symbol, 3))) == 36
        >>> indexed_symbol =  symdef(rank=4, symbol='M', dimension=3, symmetry='sym023')
        >>> assert len(set(repeat(flatten, indexed_symbol, 3))) == 30
        >>> indexed_symbol =  symdef(rank=4, symbol='M', dimension=3, symmetry='sym03_sym12')
        >>> assert len(set(repeat(flatten, indexed_symbol, 3))) == 36
        >>> indexed_symbol =  symdef(rank=4, symbol='M', dimension=3, symmetry='sym123')
        >>> assert len(set(repeat(flatten, indexed_symbol, 3))) == 30

        >>> indexed_symbol =  symdef(rank=4, symbol='M', dimension=3, symmetry='sym0123')
        >>> assert len(set(repeat(flatten, indexed_symbol, 3))) == 15

        >>> indexed_symbol =  symdef(rank=2, symbol='M', dimension=3, symmetry='anti01')
        >>> assert len(set(map(abs, repeat(flatten, indexed_symbol, 1))).difference({0})) == 3
        >>> indexed_symbol =  symdef(rank=3, symbol='M', dimension=3, symmetry='anti012')
        >>> assert len(set(map(abs, repeat(flatten, indexed_symbol, 2))).difference({0})) == 1
        >>> indexed_symbol =  symdef(rank=4, symbol='M', dimension=3, symmetry='anti0123')
        >>> assert len(set(map(abs, repeat(flatten, indexed_symbol, 3))).difference({0})) == 0
    """
    if not dimension or dimension == -1:
        dimension = 3
    if symbol is not None:
        if not isinstance(symbol, str) or not re.match(r'[\w_]', symbol):
            raise ValueError('symbol must be an alphabetic string')
    if dimension is not None:
        if not isinstance(dimension, int) or dimension <= 0:
            raise ValueError('dimension must be a positive integer')
    indexed_symbol = _init(rank * [dimension], symbol)
    if symmetry: return _symmetrize(rank, indexed_symbol, symmetry, dimension)
    return indexed_symbol

def _init(shape, symbol, index=None):
    if isinstance(shape, int):
        shape = [shape]
    if not index: index = []
    iterable = [Symbol(symbol + ''.join(str(n) for n in index + [i]))
        if symbol else sympify(0) for i in range(shape[0])]
    if len(shape) > 1:
        for i in range(shape[0]):
            iterable[i] = _init(shape[1:], symbol, index + [i])
    return iterable

def _symmetrize(rank, indexed_symbol, symmetry, dimension):
    if rank == 1:
        if symmetry == 'nosym': return indexed_symbol
        raise IndexedSymbolError('cannot symmetrize indexed symbol of rank 1')
    if rank == 2:
        indexed_symbol = _symmetrize_rank2(indexed_symbol, symmetry, dimension)
    elif rank == 3:
        indexed_symbol = _symmetrize_rank3(indexed_symbol, symmetry, dimension)
    elif rank == 4:
        indexed_symbol = _symmetrize_rank4(indexed_symbol, symmetry, dimension)
    else: raise IndexedSymbolError('unsupported rank for indexed symbol')
    return indexed_symbol

def _symmetrize_rank2(indexed_symbol, symmetry, dimension):
    for sym in symmetry.split('_'):
        sign = 1 if sym[:3] == 'sym' else -1
        for i, j in product(range(dimension), repeat=2):
            if sym[-2:] == '01':
                if j < i: indexed_symbol[i][j] = sign*indexed_symbol[j][i]
                elif i == j and sign < 0: indexed_symbol[i][j] = 0
            elif sym == 'nosym': pass
            else: raise IndexedSymbolError('unsupported symmetry option \'' + sym + '\'')
    return indexed_symbol

def _symmetrize_rank3(indexed_symbol, symmetry, dimension):
    symmetry_, symmetry = symmetry, []
    for sym in symmetry_.split('_'):
        index = 3 if sym[:3] == 'sym' else 4
        if len(sym[index:]) == 3:
            prefix = sym[:index]
            symmetry.append(prefix + sym[index:(index + 2)])
            symmetry.append(prefix + sym[(index + 1):(index + 3)])
        else: symmetry.append(sym)
    for sym in (symmetry[k] for n in range(len(symmetry), 0, -1) for k in range(n)):
        sign = 1 if sym[:3] == 'sym' else -1
        for i, j, k in product(range(dimension), repeat=3):
            if sym[-2:] == '01':
                if j < i: indexed_symbol[i][j][k] = sign*indexed_symbol[j][i][k]
                elif i == j and sign < 0: indexed_symbol[i][j][k] = 0
            elif sym[-2:] == '02':
                if k < i: indexed_symbol[i][j][k] = sign*indexed_symbol[k][j][i]
                elif i == k and sign < 0: indexed_symbol[i][j][k] = 0
            elif sym[-2:] == '12':
                if k < j: indexed_symbol[i][j][k] = sign*indexed_symbol[i][k][j]
                elif j == k and sign < 0: indexed_symbol[i][j][k] = 0
            elif sym == 'nosym': pass
            else: raise IndexedSymbolError('unsupported symmetry option \'' + sym + '\'')
    return indexed_symbol

def _symmetrize_rank4(indexed_symbol, symmetry, dimension):
    symmetry_, symmetry = symmetry, []
    for sym in symmetry_.split('_'):
        index = 3 if sym[:3] == 'sym' else 4
        if len(sym[index:]) in (3, 4):
            prefix = sym[:index]
            symmetry.append(prefix + sym[index:(index + 2)])
            symmetry.append(prefix + sym[(index + 1):(index + 3)])
            if len(sym[index:]) == 4:
                symmetry.append(prefix + sym[(index + 2):(index + 4)])
        else: symmetry.append(sym)
    for sym in (symmetry[k] for n in range(len(symmetry), 0, -1) for k in range(n)):
        sign = 1 if sym[:3] == 'sym' else -1
        for i, j, k, l in product(range(dimension), repeat=4):
            if sym[-2:] == '01':
                if j < i: indexed_symbol[i][j][k][l] = sign*indexed_symbol[j][i][k][l]
                elif i == j and sign < 0: indexed_symbol[i][j][k][l] = 0
            elif sym[-2:] == '02':
                if k < i: indexed_symbol[i][j][k][l] = sign*indexed_symbol[k][j][i][l]
                elif i == k and sign < 0: indexed_symbol[i][j][k][l] = 0
            elif sym[-2:] == '03':
                if l < i: indexed_symbol[i][j][k][l] = sign*indexed_symbol[l][j][k][i]
                elif i == l and sign < 0: indexed_symbol[i][j][k][l] = 0
            elif sym[-2:] == '12':
                if k < j: indexed_symbol[i][j][k][l] = sign*indexed_symbol[i][k][j][l]
                elif j == k and sign < 0: indexed_symbol[i][j][k][l] = 0
            elif sym[-2:] == '13':
                if l < j: indexed_symbol[i][j][k][l] = sign*indexed_symbol[i][l][k][j]
                elif j == l and sign < 0: indexed_symbol[i][j][k][l] = 0
            elif sym[-2:] == '23':
                if l < k: indexed_symbol[i][j][k][l] = sign*indexed_symbol[i][j][l][k]
                elif k == l and sign < 0: indexed_symbol[i][j][k][l] = 0
            elif sym == 'nosym': pass
            else: raise IndexedSymbolError('unsupported symmetry option \'' + sym + '\'')
    return indexed_symbol

if __name__ == "__main__":
    import doctest
    sys.exit(doctest.testmod()[0])
