# -*- coding: utf-8 -*-

from setuptools import setup, find_packages


with open("README.rst") as f:
    readme = f.read()

with open("LICENSE") as f:
    license = f.read()

setup(
    name="mnitimescales",
    version="0.1.0",
    description="Analysis of neural timescales in the MNI dataset",
    long_description=readme,
    author="Riccardo Cusinato",
    author_email="riccardo.cusinato@unibe.ch",
    url="https://github.com/cusinatr/MNI-Analysis",
    license=license,
    packages=find_packages(include=("mnitimescales")),
)
