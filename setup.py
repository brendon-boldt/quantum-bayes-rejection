#!/usr/bin/env python

from distutils.core import setup

setup(
    name="qbr",
    version="0.1",
    description="Conditioned rejection sampling for quantum Bayesian networks",
    author="The Wolf Gang",
    packages=["qbr"],
    license="GNU GPLv3",
    install_requires=["qiskit>=0.25.1"],
)
