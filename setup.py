try:
    from setuptools import setup
except ImportError:
    from distutils.core import setup

from distutils.extension import Extension
from Cython.Build import cythonize

import numpy as np

my_modules=cythonize('pysparselp/*.pyx',annotate=True)

libname="pysparselp"
setup(
name = libname,
version="0.1",
packages=         ['pysparselp'],
ext_modules = my_modules,  # additional source file(s)),
include_dirs=[ np.get_include()],
install_requires=['numpy','scipy','scikits.sparse']
)

