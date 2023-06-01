""" NRPyLaTeX Exceptions and Warnings """
# Author: Ken Sible
# Email:  ksible *at* outlook *dot* com

import warnings

class NRPyLaTeXError(Exception):

    def __init__(self, message, sentence=None, position=None):
        if position is not None:
            length = 0
            for _, substring in enumerate(sentence.split('\n')):
                if position - length <= len(substring):
                    sentence = substring.lstrip()
                    position += len(sentence) - len(substring) - length
                    break
                length += len(substring) + 1
            padding = (len(self.__class__.__name__) + position + 2) * ' '
            super(NRPyLaTeXError, self).__init__('%s\n%s^\n' % (sentence, padding) + message)
        else: super(NRPyLaTeXError, self).__init__(message)

class NamespaceError(Exception):
    """ Illegal Namespace Import """

class OverrideWarning(UserWarning):
    """ Overridden Namespace Variable """

def _formatwarning(message, category, filename=None, lineno=None, file=None, line=None):
    return '%s: %s\n' % (category.__name__, message)
warnings.formatwarning = _formatwarning
warnings.simplefilter('always', OverrideWarning)
