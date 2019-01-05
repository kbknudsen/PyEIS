import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="PyEIS",
    version="1.0.0",
    author="Kristian B. Knudsen",
    author_email="kknu@berkeley.edu",
    description="A Python-based Electrochemical Impedance Spectroscopy simulator and analyzer",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/kbknudsen/PyEIS",
    packages=setuptools.find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: Apache License, Version 2.0 (Apache-2.0)",
        "Operating System :: OS Independent",
    ],
)