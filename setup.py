#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import sys


try:
    from setuptools import setup
except ImportError:
    from distutils.core import setup


readme = open('README.rst').read()
history = open('HISTORY.rst').read().replace('.. :changelog:', '')

requirements = [
    "scipy", "numpy", "pandas"
]

test_requirements = [
   # "numpy", "pandas"
]

setup(
    name='mca',
    version='1.0',
    description='Multiple correspondence analysis with pandas DataFrames',
    long_description=readme + '\n\n' + history,
    author='Emre Safak',
    author_email='misteremre@yahoo.com',
    url='https://github.com/esafak/mca',
    download_url = 'https://github.com/esafak/mca/tarball/master',
    py_modules=['mca'],
    package_dir={'': 'src'},
    include_package_data=True,
    install_requires=requirements,
    license="BSD",
    zip_safe=False,
    keywords=['mca', 'statistics'],
    classifiers=[
        'Development Status :: 3 - Alpha',
        'Intended Audience :: Developers',
        'License :: OSI Approved :: BSD License',
        'Natural Language :: English',
        'Topic :: Scientific/Engineering :: Information Analysis',
        'Topic :: Scientific/Engineering :: Mathematics',
        "Programming Language :: Python :: 2",
        'Programming Language :: Python :: 2.6',
        'Programming Language :: Python :: 2.7',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.3',
        'Programming Language :: Python :: 3.4',
    ],
    test_suite='tests',
    tests_require=test_requirements
)