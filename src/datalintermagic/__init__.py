"""IPython magic for DataLinter"""
from .datalintermagic import DataLinterMagic


def load_ipython_extension(ipython):
    ipython.register_magics(DataLinterMagic)
