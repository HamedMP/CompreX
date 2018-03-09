from __future__ import print_function
import sys
from setuptools import setup, find_packages

with open('requirements.txt') as f:
    INSTALL_REQUIRES = [l.strip() for l in f.readlines() if l]


try:
    import numpy
except ImportError:
    print('numpy is required during installation')
    sys.exit(1)

setup(name='comprex',
      version='1.0.0',
      description='A parameter-free anomaly detection using pattern-based compression',
      author='Hamed Mohammadpour',
      packages=find_packages(),
      install_requires=INSTALL_REQUIRES,
      author_email='hamedmp2012@gmail.com',
      )
