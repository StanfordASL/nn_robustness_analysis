from setuptools import setup, find_packages

setup(
    name="nn_closed_loop",
    version="0.0.1",
    install_requires=[
        "torch",
        "matplotlib",
        "pandas==1.5.3",
        "nn_partition",
        "tabulate",
    ],
    packages=find_packages(),
)
