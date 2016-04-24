#run python setup.py build_ext --inplace

from distutils.core import setup
#from distutils.extension import Extension
from Cython.Distutils import build_ext
# from Cython.Build import cythonize
import numpy as np
from Cython.Build import cythonize


my_modules=cythonize('*.pyx',annotate=True)

libname="pyspraselp"
setup(
name = libname,
version="0.1",
packages=         ['pyspraselp'],
ext_modules = my_modules,  # additional source file(s)),
include_dirs=[ np.get_include()],
)
