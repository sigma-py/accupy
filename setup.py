import sys

import setuptools
from setuptools import Extension, find_packages, setup
from setuptools.command.build_ext import build_ext


# https://github.com/pybind/python_example/
class get_pybind_include:
    def __str__(self):
        import pybind11

        return pybind11.get_include()


ext_modules = [
    Extension(
        "_accupy",
        ["src/pybind11.cpp"],
        language="c++",
        include_dirs=["/usr/include/eigen3/", get_pybind_include()],
    )
]


# cf http://bugs.python.org/issue26689
def has_flag(compiler, flagname):
    """Return a boolean indicating whether a flag name is supported on
    the specified compiler.
    """
    import tempfile
    import os

    with tempfile.NamedTemporaryFile("w", suffix=".cpp", delete=False) as f:
        f.write("int main (int argc, char **argv) { return 0; }")
        fname = f.name
    try:
        compiler.compile([fname], extra_postargs=[flagname])
    except setuptools.distutils.errors.CompileError:
        return False
    finally:
        try:
            os.remove(fname)
        except OSError:
            pass
    return True


def cpp_flag(compiler):
    flags = ["-std=c++17", "-std=c++14", "-std=c++11"]

    for flag in flags:
        if has_flag(compiler, flag):
            return flag

    raise RuntimeError("Unsupported compiler -- at least C++11 support is needed!")


class BuildExt(build_ext):
    c_opts = {
        "msvc": ["/EHsc"],
        "unix": [],
    }
    l_opts = {
        "msvc": [],
        "unix": [],
    }

    if sys.platform == "darwin":
        darwin_opts = ["-stdlib=libc++", "-mmacosx-version-min=10.7"]
        c_opts["unix"] += darwin_opts
        l_opts["unix"] += darwin_opts

    def build_extensions(self):
        ct = self.compiler.compiler_type
        opts = self.c_opts.get(ct, [])
        link_opts = self.l_opts.get(ct, [])
        if ct == "unix":
            opts.append(cpp_flag(self.compiler))
            if has_flag(self.compiler, "-fvisibility=hidden"):
                opts.append("-fvisibility=hidden")

        for ext in self.extensions:
            ext.define_macros = [
                ("VERSION_INFO", '"{}"'.format(self.distribution.get_version()))
            ]
            ext.extra_compile_args = opts
            ext.extra_link_args = link_opts
        build_ext.build_extensions(self)


setup(
    name="accupy",
    version="0.3.1",
    packages=find_packages(),
    ext_modules=ext_modules,
    url="https://github.com/nschloe/accupy",
    author="Nico Schlömer",
    author_email="nico.schloemer@gmail.com",
    # importlib_metadata can be removed when we support Python 3.8+ only
    install_requires=[
        "importlib_metadata",
        "mpmath",
        "numpy",
        "pybind11 >= 2.5.0",
        "pyfma",
    ],
    setup_requires=["pybind11 >= 2.5.0"],
    cmdclass={"build_ext": BuildExt},
    python_requires=">=3.6",
    description="Accurate sums and dot products for Python",
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
    license="GPLv3+",
    classifiers=[
        "Development Status :: 4 - Beta",
        "License :: OSI Approved :: GNU General Public License v3 or later (GPLv3+)",
        "Operating System :: OS Independent",
        "Programming Language :: Python",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.6",
        "Programming Language :: Python :: 3.7",
        "Programming Language :: Python :: 3.8",
        "Topic :: Scientific/Engineering",
        "Topic :: Scientific/Engineering :: Mathematics",
    ],
)
