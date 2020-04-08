"""Setup script"""

try:
    from setuptools import setup
except ImportError:
    from distutils.core import setup

from Cython.Build import cythonize

import numpy as np

my_modules = cythonize("pysparselp/*.pyx", annotate=True)

libname = "pysparselp"
setup(
    name=libname,
    version="0.0.1",
    author="Martin de La Gorce",
    author_email="martin.delagorce@gmail.com",
    description="Python algorithms to solve linear programming problems with with sparse matrices",
    packages=["pysparselp"],
    license="MIT",
    ext_modules=my_modules,  # additional source file(s)),
    include_dirs=[np.get_include()],
    package_data={"pysparselp": ["*.pyx"]},
    install_requires=["numpy", "scipy"],
)
