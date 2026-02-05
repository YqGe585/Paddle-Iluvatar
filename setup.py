"""
Setup script for Paddle-Iluvatar GPU adapter
"""
from setuptools import setup, find_packages, Extension
from setuptools.command.build_ext import build_ext
import sys
import os

__version__ = '0.1.0'

class CMakeExtension(Extension):
    def __init__(self, name, sourcedir=''):
        Extension.__init__(self, name, sources=[])
        self.sourcedir = os.path.abspath(sourcedir)

class CMakeBuild(build_ext):
    def run(self):
        for ext in self.extensions:
            self.build_extension(ext)

    def build_extension(self, ext):
        print(f"Building extension: {ext.name}")

setup(
    name='paddle-iluvatar',
    version=__version__,
    author='Paddle-Iluvatar Contributors',
    description='Iluvatar GPU adapter for PaddlePaddle',
    long_description=open('README.md').read(),
    long_description_content_type='text/markdown',
    packages=find_packages(where='python'),
    package_dir={'': 'python'},
    ext_modules=[CMakeExtension('paddle_iluvatar._C')],
    cmdclass=dict(build_ext=CMakeBuild),
    install_requires=[
        'numpy>=1.19.0',
    ],
    extras_require={
        'dev': [
            'pytest>=6.0',
            'pytest-cov>=2.10',
        ],
    },
    python_requires='>=3.6',
    classifiers=[
        'Development Status :: 3 - Alpha',
        'Intended Audience :: Developers',
        'License :: OSI Approved :: Apache Software License',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.6',
        'Programming Language :: Python :: 3.7',
        'Programming Language :: Python :: 3.8',
        'Programming Language :: Python :: 3.9',
        'Programming Language :: Python :: 3.10',
    ],
)
