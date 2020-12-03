# https://github.com/pybind/python_example
from pybind11.setup_helpers import Pybind11Extension, build_ext
from setuptools import setup

ext_modules = [
    Pybind11Extension(
        "_accupy",
        ["src/pybind11.cpp"],
        include_dirs=["/usr/include/eigen3/"],
    )
]

if __name__ == "__main__":
    setup(
        cmdclass={"build_ext": build_ext},
        ext_modules=ext_modules,
        zip_safe=False,
    )
