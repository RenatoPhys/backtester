from setuptools import setup, find_packages

setup(
    name="futures_backtester",  # Modificado de "backtester" para "futures_backtester"
    version="0.1.0",
    packages=find_packages(),
    install_requires=[
        "numpy",
        "pandas",
        "matplotlib", 
        "pyarrow",
        "numba"
    ],
)