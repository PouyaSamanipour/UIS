from setuptools import setup, find_packages

setup(
    name="UIS_Module",  # Choose a suitable package name
    version="0.1",
    packages=find_packages(),
    install_requires=[
        "numpy",
        "numba",
        "torch",
        "scipy",
        "matplotlib",
        "pandas", 
        "pycddlib"
    ],
    author="Pouya Samanipour",
    author_email="psa254@uky.edu",
    description="Union of Invariant sets for Safety verification",
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
    url="https://github.com/PouyaSamanipour/UIS",
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.6",
)
