#!/usr/bin/env python3

# Set this to True to enable building extensions using Cython.
# Set it to False to build extensions from the C file (that
# was previously created using Cython).
# Set it to 'auto' to build with Cython if available, otherwise
# from the C file.
USE_CYTHON = False

# https://stackoverflow.com/questions/49471084/prepare-c-based-cython-package-to-publish-on-pypi

import sys
import numpy
from setuptools import setup, Extension, find_packages
# from setuptools import Extension
from Cython.Distutils import build_ext
from Cython.Build import cythonize
import os.path

ext = '.pyx' if USE_CYTHON else '.c'
# ext_modules = []
# cmdclass = {}

# def no_cythonize(extensions, **_ignore):
#     for extension in extensions:
#         sources = []
#         for sfile in extension.sources:
#             path, ext = os.path.splitext(sfile)
#             if ext in ('.pyx', '.py'):
#                 if extension.language == 'c++':
#                     ext = '.cpp'
#                 else:
#                     ext = '.c'
#                 sfile = path + ext
#             sources.append(sfile)
#         extension.sources[:] = sources
#     return extensions

cmdclass = { }
# ext_modules = [ ]
# cmdclass.update({ 'build_ext': build_ext })

if USE_CYTHON:
    ext_modules = [Extension("boostdiff.differential_trees.arandom", ["boostdiff/differential_trees/arandom.pyx"]),
                 Extension("boostdiff.differential_trees.splitter", ["boostdiff/differential_trees/splitter.pyx"]),
                 Extension("boostdiff.differential_trees.utils", ["boostdiff/differential_trees/utils.pyx"]),
                 Extension("boostdiff.differential_trees.diff_tree", ["boostdiff/differential_trees/diff_tree.pyx"]),]
    
    cmdclass = { 'build_ext': build_ext }
else:
    ext_modules = [Extension("boostdiff.differential_trees.arandom", ["boostdiff/differential_trees/arandom.c"]),
                 Extension("boostdiff.differential_trees.splitter", ["boostdiff/differential_trees/splitter.c"]),
                 Extension("boostdiff.differential_trees.utils", ["boostdiff/differential_trees/utils.c"]),
                 Extension("boostdiff.differential_trees.diff_tree", ["boostdiff/differential_trees/diff_tree.c"]),]

if sys.version_info[0] == 2:
    raise Exception('Python 2.x is not supported')

# ext_modules=cythonize(["differential_trees/*.pyx"]), include_dirs=[numpy.get_include()],

setup(
    name='boostdiff',
    version="1.0",
    description='Boosted differential trees algorithm',
    author='Gihanna Galindez',
    author_email='gihanna.galindez@plri.de',
    url='http://github.com/gihannagalindez/boostdiff_inference',
    packages=find_packages(),
    package_dir={
        'boostdiff':'boostdiff',
    },
    cmdclass = cmdclass,
    ext_modules=ext_modules,
    include_dirs=[numpy.get_include()],
    
    license="GPLv3",
    classifiers=[
            "Programming Language :: Python :: 3",
            "Programming Language :: Python :: 3.7",
            "Programming Language :: Python :: 3.8",
            "License :: OSI Approved :: GNU General Public License v3 (GPLv3)",
            "Topic :: Scientific/Engineering :: Bio-Informatics",
          ],
    install_requires=[
          'numpy',
          'pandas',
          'matplotlib',
      ],
    keywords='differential network inference',
)