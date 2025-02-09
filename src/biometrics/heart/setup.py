# python setup.py build_ext --build-lib /Users/ds/main/8sleep_biometrics/src/biometrics/heart/ --inplace
from setuptools import setup
from Cython.Build import cythonize
import numpy

setup(
    ext_modules=cythonize("peakdetection_cython.pyx", language_level="3"),
    include_dirs=[numpy.get_include()],  # Add NumPy headers
)
