from nrpylatex.extension.parse_magic import ParseMagic

def load_ipython_extension(ipython):
    ipython.register_magics(ParseMagic)
