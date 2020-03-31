from setuptools import Extension, find_packages, setup


class get_pybind_include:
    def __init__(self, user=False):
        self.user = user

    def __str__(self):
        import pybind11

        return pybind11.get_include(self.user)


ext_modules = [
    Extension(
        "_accupy",
        ["src/pybind11.cpp"],
        language="c++",
        include_dirs=[
            "/usr/include/eigen3/",
            get_pybind_include(),
            get_pybind_include(user=True),
        ],
    )
]


setup(
    name="accupy",
    version="0.2.0",
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
        "pybind11 >= 2.2",
        "pyfma",
    ],
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
