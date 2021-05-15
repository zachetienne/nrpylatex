""" Module to Define Indexed Symbol(s) """
# Author: Ken Sible
# Email:  ksible *at* outlook *dot* com

from nrpylatex.core.functional import pipe, repeat, flatten, product
import sympy as sp, re, sys

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
    iterable = [sp.Symbol(symbol + ''.join(str(n) for n in index + [i]))
        if symbol else sp.sympify(0) for i in range(shape[0])]
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

class IndexedSymbolError(Exception):
    """ Invalid Indexed Symbol """

if __name__ == "__main__":
    import doctest
    sys.exit(doctest.testmod()[0])
