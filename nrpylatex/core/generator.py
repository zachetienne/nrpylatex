""" NRPyLaTeX Code Generator """
# Author: Ken Sible
# Email:  ksible *at* outlook *dot* com

from nrpylatex.utils.structures import ExprTree, IndexedSymbol
from nrpylatex.utils.exceptions import NRPyLaTeXError
from nrpylatex.utils.functional import uniquify, flatten
from sympy import Function, Derivative, Symbol, Add, srepr
import math, re

class Generator:

    def __init__(self, parser):
        self._namespace = parser._namespace
        self._property = parser._property

    def generate(self, LHS, RHS, impsum=True):
        # perform implied summation on indexed expression
        LHS_RHS, dimension = self.expand_summation(LHS, RHS, impsum)
        if self._property['debug']:
            lineno = '[%d]' % self._property['debug']
            print('%s Python' % (len(lineno) * ' '))
            print('%s   %s\n' % (len(lineno) * ' ', LHS_RHS))
            self._property['debug'] += 1

        global_env = dict(self._namespace)
        for key in global_env:
            if isinstance(global_env[key], IndexedSymbol):
                global_env[key] = global_env[key].structure
            if isinstance(global_env[key], Function('Constant')):
                global_env[key] = global_env[key].args[0]
        global_env['coord'] = self._property['coord']

        # evaluate every implied summation and update namespace
        exec('from sympy import *', global_env)
        try: exec(LHS_RHS, global_env)
        except IndexError:
            raise GeneratorError('index out of range; change loop/summation range')

        return global_env, dimension

    def expand_summation(self, LHS, RHS, impsum=True):
        tree, indexing = ExprTree(LHS), []
        for subtree in tree.preorder():
            subexpr = subtree.expr
            if subexpr.func == Function('Tensor'):
                for index, position in IndexedSymbol.indexing(subexpr):
                    if re.match(r'[a-zA-Z]+(?:_[0-9]+)?', str(index)):
                        indexing.append((index, position))
            elif subexpr.func == Derivative:
                for index, _ in subexpr.args[1:]:
                    if index not in self._property['coord']:
                        if re.match(r'[a-zA-Z]+(?:_[0-9]+)?', str(index)):
                            indexing.append((index, 'D'))

        symbol_LHS = IndexedSymbol(LHS).symbol
        # construct a tuple list of every LHS free index
        free_index_LHS = self.separate_indexing(indexing, symbol_LHS, impsum)[0] if impsum \
            else uniquify([(str(idx), pos) for idx, pos in indexing])
        # construct a tuple list of every RHS free index
        free_index_RHS = []

        iterable = RHS.args if RHS.func == Add else [RHS]
        LHS, RHS = IndexedSymbol(LHS).array_format(LHS), srepr(RHS)
        for element in iterable:
            index_range = self._property['index'].copy()
            original = srepr(element)
            if original[0] == '-':
                original = original[1:]
            modified = original
            indexing = []
            tree = ExprTree(element)

            for subtree in tree.preorder():
                subexpr = subtree.expr
                if subexpr.func == Function('Tensor'):
                    symbol = str(subexpr.args[0])
                    dimension = self._namespace[symbol].dimension
                    for index in subexpr.args[1:]:
                        if str(index) in self._property['index']:
                            dimension = self._property['index'][str(index)]
                        if str(index) in index_range and dimension != index_range[str(index)]:
                            raise GeneratorError('inconsistent loop/summation range for index \'%s\'' %
                                index, self.scanner.sentence)
                        index_range[str(index)] = dimension
                    function = IndexedSymbol(subexpr).array_format(subexpr)
                    modified = modified.replace(srepr(subexpr), function)
                    for index, position in IndexedSymbol.indexing(subexpr):
                        if re.match(r'[a-zA-Z]+(?:_[0-9]+)?', str(index)):
                            indexing.append((index, position))
                elif subexpr.func == Function('Constant'):
                    constant = str(subexpr.args[0])
                    modified = modified.replace(srepr(subexpr), constant)
                elif subexpr.func == Derivative:
                    argument = subexpr.args[0]
                    derivative = 'diff(' + srepr(argument)
                    symbol = str(argument.args[0])
                    dimension = self._namespace[symbol].dimension
                    for index, order in subexpr.args[1:]:
                        if str(index) in self._property['index']:
                            dimension = self._property['index'][str(index)]
                        if str(index) in index_range and dimension != index_range[str(index)]:
                            raise GeneratorError('inconsistent loop/summation range for index \'%s\'' %
                                index, self.scanner.sentence)
                        index_range[str(index)] = dimension
                        if index not in self._property['coord']:
                            derivative += ', (coord[%s], %s)' % (index, order)
                            if re.match(r'[a-zA-Z]+(?:_[0-9]+)?', str(index)):
                                indexing.append((index, 'D'))
                        else: derivative += ', (%s, %s)' % (index, order)
                    derivative += ')'
                    modified = modified.replace(srepr(subexpr), derivative)
                    tmp = srepr(subexpr).replace(srepr(argument), IndexedSymbol(argument).array_format(argument))
                    modified = modified.replace(tmp, derivative)

            if impsum:
                free_index, bound_index = self.separate_indexing(indexing, symbol_LHS, impsum)
                free_index_RHS.append(free_index)
                # generate implied summation over every bound index
                for idx in bound_index:
                    modified = 'sum(%s for %s in range(%d))' % (modified, idx, index_range[idx])
            else:
                free_index_RHS.append(indexing)
            RHS = RHS.replace(original, modified)

        if impsum:
            unique_LHS = uniquify([(str(idx), pos) for idx, pos in free_index_LHS])
            unique_RHS = uniquify([(str(idx), pos) for idx, pos in flatten(free_index_RHS)])
            for idx, pos in unique_LHS + unique_RHS:
                if ((idx, pos) in unique_LHS) != ((idx, pos) in unique_RHS):
                    # raise exception upon violation of the following rule:
                    # a free index must appear in every term with the same
                    # position and cannot be summed over in any term
                    raise GeneratorError('unbalanced free index \'%s\' in %s' % (idx, symbol_LHS))
        else:
            unique_LHS = uniquify([str(idx) for idx, _ in free_index_LHS])
            unique_RHS = uniquify([str(idx) for idx, _ in flatten(free_index_RHS)])
            for idx in unique_RHS:
                if idx not in unique_LHS:
                    # raise exception upon violation of the following rule:
                    # every index on the RHS must appear at least once on
                    # the LHS with the noimpsum annotation applied
                    raise GeneratorError('unbalanced index \'%s\' in %s' % (idx, symbol_LHS))

        # generate tensor instantiation with implied summation
        if symbol_LHS in self._namespace:
            equation = len(free_index_LHS) * '    ' + '%s = %s' % (LHS, RHS)
            for i, (idx, _) in enumerate(reversed(free_index_LHS)):
                indent = len(free_index_LHS) - (i + 1)
                equation = indent * '    ' + 'for %s in range(%d):\n' % (idx, index_range[idx]) + equation
            equation = [equation]
        else:
            for idx, _ in reversed(free_index_LHS):
                RHS = '[%s for %s in range(%d)]' % (RHS, idx, index_range[idx])
            equation = [LHS.split('[')[0], RHS]

        dimension_LHS = None
        if free_index_LHS:
            if len(uniquify(index_range[index] for index, _ in free_index_LHS)) > 1:
                raise GeneratorError('cannot infer dimension of \'%s\'' % symbol_LHS, self.scanner.sentence)
            index, _ = free_index_LHS[0]
            dimension_LHS = index_range[index]

        # shift tensor indexing forward whenever dimension > upper bound
        for subtree in tree.preorder():
            subexpr = subtree.expr
            if subexpr.func == Function('Tensor'):
                symbol = str(subexpr.args[0])
                dimension = self._namespace[symbol].dimension
                tensor = IndexedSymbol(subexpr, dimension)
                indexing = IndexedSymbol.indexing(subexpr)
                for index in subexpr.args[1:]:
                    if str(index) in self._property['index']:
                        upper_bound = self._property['index'][str(index)]
                        if dimension > upper_bound:
                            shift = dimension - upper_bound
                            for i, (idx, pos) in enumerate(indexing):
                                if str(idx) == str(index):
                                    indexing[i] = ('%s + %s' % (idx, shift), pos)
                equation[-1] = equation[-1].replace(tensor.array_format(subexpr), tensor.array_format(indexing))

        return ' = '.join(equation), dimension_LHS

    @staticmethod
    def separate_indexing(indexing, symbol_LHS, impsum=True):
        free_index, bound_index = [], []
        indexing = [(str(idx), pos) for idx, pos in indexing]
        # iterate over every unique index in the subexpression
        for index in uniquify([idx for idx, _ in indexing]):
            count = U = D = 0; index_tuple = []
            # count index occurrence and position occurrence
            for index_, position in indexing:
                if index_ == index:
                    index_tuple.append((index_, position))
                    if position == 'U': U += 1
                    if position == 'D': D += 1
                    count += 1
            # identify every bound index on the RHS
            if count > 1:
                if impsum and (count != 2 or U != D):
                    # raise exception upon violation of the following rule:
                    # a bound index must appear exactly once as a superscript
                    # and exactly once as a subscript in any single term
                    raise GeneratorError('illegal bound index \'%s\' in %s' % (index, symbol_LHS))
                bound_index.append(index)
            # identify every free index on the RHS
            else: free_index.extend(index_tuple)
        return uniquify(free_index), bound_index

    @staticmethod
    def generate_metric(symbol, dimension, diacritic, suffix):
        latex_config = ''
        if 'U' in symbol:
            prefix = r'\epsilon_{' + ' '.join('i_' + str(i) for i in range(1, 1 + dimension)) + '} ' + \
                    r'\epsilon_{' + ' '.join('j_' + str(i) for i in range(1, 1 + dimension)) + '} '
            det_latex = prefix + ' '.join(r'\mathrm{{{symbol}}}^{{i_{n} j_{n}}}'.format(symbol=symbol[:-2], n=i) for i in range(1, 1 + dimension))
            inv_latex = prefix + ' '.join(r'\mathrm{{{symbol}}}^{{i_{n} j_{n}}}'.format(symbol=symbol[:-2], n=i) for i in range(2, 1 + dimension))
            latex_config += r"""
    \mathrm{{{symbol}det}} = \frac{{1}}{{({dimension})({factorial})}} {det_latex} \\
    \mathrm{{{symbol}}}_{{i_1 j_1}} = \frac{{1}}{{{factorial}}} \mathrm{{{symbol}det}}^{{{{-1}}}} ({inv_latex})""" \
                .format(symbol=symbol[:-2], inv_symbol=symbol.replace('U', 'D'), dimension=dimension,
                    factorial=math.factorial(dimension - 1), det_latex=det_latex, inv_latex=inv_latex)
            latex_config += '\n' + r"% assign {symbol}det --dim {dimension}".format(symbol=symbol[:-2], dimension=dimension)
            if suffix:
                latex_config += '\n' + r"% assign {symbol}det {inv_symbol} --suffix {suffix}" \
                    .format(suffix=suffix, symbol=symbol[:-2], inv_symbol=symbol.replace('U', 'D'))
        else:
            prefix = r'\epsilon^{' + ' '.join('i_' + str(i) for i in range(1, 1 + dimension)) + '} ' + \
                    r'\epsilon^{' + ' '.join('j_' + str(i) for i in range(1, 1 + dimension)) + '} '
            det_latex = prefix + ' '.join(r'\mathrm{{{symbol}}}_{{i_{n} j_{n}}}'.format(symbol=symbol[:-2], n=i) for i in range(1, 1 + dimension))
            inv_latex = prefix + ' '.join(r'\mathrm{{{symbol}}}_{{i_{n} j_{n}}}'.format(symbol=symbol[:-2], n=i) for i in range(2, 1 + dimension))
            latex_config += r"""
    \mathrm{{{symbol}det}} = \frac{{1}}{{({dimension})({factorial})}} {det_latex} \\
    \mathrm{{{symbol}}}^{{i_1 j_1}} = \frac{{1}}{{{factorial}}} \mathrm{{{symbol}det}}^{{{{-1}}}} ({inv_latex})""" \
                .format(symbol=symbol[:-2], inv_symbol=symbol.replace('D', 'U'), dimension=dimension,
                    factorial=math.factorial(dimension - 1), det_latex=det_latex, inv_latex=inv_latex)
            latex_config += '\n' + r"% assign {symbol}det --dim {dimension}".format(symbol=symbol[:-2], dimension=dimension)
            if suffix:
                latex_config += '\n' + r"% assign {symbol}det {inv_symbol} --suffix {suffix}" \
                    .format(suffix=suffix, symbol=symbol[:-2], inv_symbol=symbol.replace('D', 'U'))
        metric = '\\mathrm{' + re.split(r'[UD]', symbol)[0] + '}'
        latex_config += '\n' + r'\mathrm{{Gamma{diacritic}}}^{{i_1}}_{{i_2 i_3}} = \frac{{1}}{{2}} {metric}^{{i_1 i_4}} (\partial_{{i_2}} {metric}_{{i_3 i_4}} + \partial_{{i_3}} {metric}_{{i_4 i_2}} - \partial_{{i_4}} {metric}_{{i_2 i_3}})'.format(metric=metric, diacritic=diacritic)
        return latex_config

    @staticmethod
    def generate_covdrv(function, covdrv_index, symbol=None, diacritic=None):
        indexing = [str(index) for index in function.args[1:]] + [str(covdrv_index)]
        idx_gen = IndexedSymbol.index_count()
        for i, index in enumerate(indexing):
            if index in indexing[:i]:
                indexing[i] = next(x for x in idx_gen if x not in indexing)
        covdrv_index = indexing[-1]
        if '_' in str(covdrv_index):
            base, subscript = str(covdrv_index).split('_')
            if len(base) > 1:
                covdrv_index = '\\%s_%s' % (base, subscript)
        elif len(str(covdrv_index)) > 1:
            covdrv_index = '\\' + str(covdrv_index)
        latex = IndexedSymbol.latex_format(Function('Tensor')(function.args[0],
            *(Symbol(i) for i in indexing[:-1])))
        LHS = ('\\%s{\\nabla}' % diacritic if diacritic else '\\nabla') + ('_{%s} %s' % (covdrv_index, latex))
        RHS = '\\partial_{%s} %s' % (covdrv_index, latex)
        for index, (_, position) in zip(indexing, IndexedSymbol.indexing(function)):
            idx_gen = IndexedSymbol.index_count()
            bound_index = next(x for x in idx_gen if x not in indexing)
            latex = IndexedSymbol.latex_format(Function('Tensor')(function.args[0],
                *(Symbol(bound_index) if i == index else Symbol(i) for i in indexing[:-1])))
            if '_' in str(index):
                base, subscript = str(index).split('_')
                if len(base) > 1:
                    index = '\\%s_%s' % (base, subscript)
            elif len(str(index)) > 1:
                index = '\\' + str(index)
            RHS += ' + ' if position == 'U' else ' - '
            RHS += '\\%s{\\mathrm{Gamma}}' % diacritic if diacritic else '\\mathrm{Gamma}'
            if position == 'U':
                RHS += '^{%s}_{%s %s} (%s)' % (index, bound_index, covdrv_index, latex)
            else:
                RHS += '^{%s}_{%s %s} (%s)' % (bound_index, index, covdrv_index, latex)
        config = (' % assign ' + symbol + ' --suffix dD\n') if symbol else ''
        return LHS + ' = ' + RHS + config

    @staticmethod
    def generate_liedrv(function, vector, weight=None):
        if len(str(vector)) > 1:
            vector = '\\mathrm{' + str(vector) + '}'
        indexing = [str(index) for index, _ in IndexedSymbol.indexing(function)]
        idx_gen = IndexedSymbol.index_count()
        for i, index in enumerate(indexing):
            if index in indexing[:i]:
                indexing[i] = next(x for x in idx_gen if x not in indexing)
        latex = IndexedSymbol.latex_format(function)
        LHS = '\\mathcal{L}_%s %s' % (vector, latex)
        bound_index = next(x for x in idx_gen if x not in indexing)
        RHS = '%s^{%s} \\partial_{%s} %s' % (vector, bound_index, bound_index, latex)
        for index, position in IndexedSymbol.indexing(function):
            latex = IndexedSymbol.latex_format(Function('Tensor')(function.args[0],
                *(Symbol(bound_index) if i == str(index) else Symbol(i) for i in indexing)))
            if '_' in str(index):
                base, subscript = str(index).split('_')
                if len(base) > 1:
                    index = '\\%s_%s' % (base, subscript)
            elif len(str(index)) > 1:
                index = '\\' + str(index)
            if position == 'U':
                RHS += ' - (\\partial_{%s} %s^{%s}) %s' % (bound_index, vector, index, latex)
            else:
                RHS += ' + (\\partial_{%s} %s^{%s}) %s' % (index, vector, bound_index, latex)
        if weight:
            latex = IndexedSymbol.latex_format(function)
            RHS += ' + (%s)(\\partial_{%s} %s^{%s}) %s' % (weight, bound_index, vector, bound_index, latex)
        return LHS + ' = ' + RHS

class GeneratorError(NRPyLaTeXError):

    def __init__(self, message, sentence=None, position=None):
        super(GeneratorError, self).__init__(message, sentence, position)
