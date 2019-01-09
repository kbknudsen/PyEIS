import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="PyEIS",
    version="1.0.1",
    author="Kristian B. Knudsen",
    author_email="kknu@berkeley.edu",
    description="A Python-based Electrochemical Impedance Spectroscopy simulator and analyzer",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/kbknudsen/PyEIS",
    packages=['PyEIS'],
    scripts=['bin/PyEIS_script.py'],
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: Apache Software License",
        "Operating System :: OS Independent",
    ],
)