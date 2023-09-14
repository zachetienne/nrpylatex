""" NRPyLaTeX Exceptions """
# Author: Ken Sible
# Email:  ksible *at* outlook *dot* com

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
            super(NRPyLaTeXError, self).__init__(message
                + '\n  %s\n%s^' % (sentence, (position + 2) * ' '))
        else:
            super(NRPyLaTeXError, self).__init__(message)

class NamespaceError(Exception):
    """ Illegal Namespace Import """

class DeprecatedWarning(Warning):
    """ Use of Deprecated Feature """
