#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Generated with pypackage cookiecutter
"""The setup script."""

from setuptools import setup, find_packages

requirements = ['Click>=6.0', "numpy>=1.17.0", "pandas>=0.25.0"]

setup_requirements = [ ]

test_requirements = [ ]

setup(
    author="Eric Horvat & Daniel Glatter",
    author_email='eric.nahuel.horvat@gmail.com',
    classifiers=[
        'Development Status :: 2 - Pre-Alpha',
        'Intended Audience :: Developers',
        'License :: OSI Approved :: GNU General Public License v3 (GPLv3)',
        'Natural Language :: English',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.6',
        'Programming Language :: Python :: 3.7',
    ],
    description="Machine Learning exercises from ITBA course",
    entry_points={
        'console_scripts': [
            'ml_tps=ml_tps.cli:cli',
        ],
    },
    install_requires=requirements,
    license="GNU General Public License v3",
    long_description= '',
    include_package_data=True,
    keywords='ml_tps',
    name='ml_tps',
    packages=find_packages(include=['ml_tps']),
    setup_requires=setup_requirements,
    url='https://github.com/amkcpu/Aprendizaje-Automatico---TPs',
    version='0.1.0',
    zip_safe=False,
)
