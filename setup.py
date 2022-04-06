"""Install packages as defined in this file into the Python environment."""
from setuptools import setup, find_packages

# The version of this tool is based on the following steps:
# https://packaging.python.org/guides/single-sourcing-package-version/
VERSION = {}

setup(
    name="chisel4ml",
    author="Jure Vreca",
    author_email="jure.vreca@ijs.si",
    url="https://github.com/jurevreca12/chisel4ml",
    description="TODO",
    version=VERSION.get("__version__", "0.0.0"),
    packages=find_packages(where=".", exclude=["tests"]),
    install_requires=[
        "setuptools>=45.0",
    ],
    classifiers=[
        "Development Status :: 1 - Planning",
        "Programming Language :: Python :: 3.0",
        "Topic :: Utilities",
    ],
)
