#!/usr/bin/env python
# -*- coding: utf-8 -*-

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
    version='1.0.4',
    description='Multiple correspondence analysis with pandas',
    long_description=readme + '\n\n' + history,
    long_description_content_type='text/x-rst',
    author='Emre Safak',
    author_email='misteremre@yahoo.com',
    url='https://github.com/esafak/mca',
    download_url = 'https://github.com/esafak/mca/tarball/master',
    py_modules=['mca'],
    package_dir={'': 'src'},
    include_package_data=True,
    install_requires=requirements,
    license = "MIT AND (Apache-2.0 OR BSD-2-Clause)",
    zip_safe=False,
    keywords=['mca', 'statistics'],
    classifiers=[
        'Development Status :: 4 - Beta',
        'Intended Audience :: Developers',
        'Intended Audience :: Science/Research',
        'License :: OSI Approved :: BSD License',
        'Natural Language :: English',
        'Topic :: Scientific/Engineering :: Information Analysis',
        'Topic :: Scientific/Engineering :: Mathematics',
        "Programming Language :: Python :: 2",
        'Programming Language :: Python :: 2.7',
        'Programming Language :: Python :: 3',
    ],
    test_suite='tests',
    tests_require=test_requirements
)