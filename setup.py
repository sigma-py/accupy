# -*- coding: utf-8 -*-
#
import os
import codecs

from setuptools import setup, find_packages, Extension

# https://packaging.python.org/single_source_version/
base_dir = os.path.abspath(os.path.dirname(__file__))
about = {}
with open(os.path.join(base_dir, 'accupy', '__about__.py'), 'rb') as f:
    # pylint: disable=exec-used
    exec(f.read(), about)


class get_pybind_include(object):
    '''Helper class to determine the pybind11 include path
    The purpose of this class is to postpone importing pybind11
    until it is actually installed, so that the ``get_include()``
    method can be invoked.
    '''
    def __init__(self, user=False):
        self.user = user

    def __str__(self):
        import pybind11
        return pybind11.get_include(self.user)


ext_modules = [Extension(
    '_accupy',
    [
        'src/pybind11.cpp'
    ],
    language='c++',
    include_dirs=[
        '/usr/include/eigen3/',
        get_pybind_include(),
        get_pybind_include(user=True)
        ],
    )]


def read(fname):
    return codecs.open(os.path.join(base_dir, fname), encoding='utf-8').read()


setup(
    name='accupy',
    version=about['__version__'],
    packages=find_packages(),
    ext_modules=ext_modules,
    url='https://github.com/nschloe/accupy',
    author=about['__author__'],
    author_email=about['__email__'],
    install_requires=[
        'mpmath',
        'numpy',
        'pipdate',
        'pybind11 >= 2.2',
        'pyfma',
        ],
    description='Accurate sums and dot products for Python',
    long_description=read('README.md'),
    long_description_content_type='text/markdown',
    license=about['__license__'],
    classifiers=[
        about['__license__'],
        about['__status__'],
        'Operating System :: OS Independent',
        'Programming Language :: Python',
        'Programming Language :: Python :: 2',
        'Programming Language :: Python :: 3',
        'Topic :: Scientific/Engineering',
        'Topic :: Scientific/Engineering :: Mathematics'
        ]
    )
