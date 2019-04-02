#from distutils.core import setup
#from Cython.Build import cythonize
import numpy
#
#setup(
#    ext_modules = cythonize("src/libgmdc.pyx", annotate=True),
#    include_dirs=[numpy.get_include()]
#)

from distutils.core import setup
from distutils.extension import Extension
from Cython.Distutils import build_ext

setup(
  name = 'GMD',
  ext_modules=[
    Extension('libgmdc',
              sources=['src/libgmdc.pyx'],
              extra_compile_args=['-O3','-ffast-math'],
              language='c')
    ],
  include_dirs=[numpy.get_include()],
  cmdclass = {'build_ext': build_ext}
)

