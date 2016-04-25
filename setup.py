try:
    from setuptools import setup
except ImportError:
    from distutils.core import setup

from distutils.extension import Extension
from Cython.Build import cythonize

import numpy as np

my_modules=cythonize('code/*.pyx',annotate=True)

libname="pyspraselp"
setup(
name = libname,
version="0.1",
packages=         ['pyspraselp'],
ext_modules = my_modules,  # additional source file(s)),
include_dirs=[ np.get_include()],
)

