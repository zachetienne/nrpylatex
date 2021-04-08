from setuptools import setup
import pathlib

HERE = pathlib.Path(__file__).parent
README = (HERE / "README.md").read_text()

setup(
    name="nrpylatex",
    version="1.0.0",
    description="LaTeX Interface to SymPy (CAS) for Numerical Relativity",
    long_description=README,
    long_description_content_type="text/markdown",
    url="https://github.com/zachetienne/nrpylatex",
    author="Ken Sible",
    author_email="ksible@outlook.com",
    license="BSD 2-Clause License",
    packages=["nrpylatex", "nrpylatex.tests"],
    install_requires=["sympy"]
)
