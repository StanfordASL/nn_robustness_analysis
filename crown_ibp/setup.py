from setuptools import setup, find_packages

setup(name='crown_ibp',
      version='1.0.0',
      install_requires=[
          'torch',
          'tensorflow',
          ],
      packages=find_packages()
)