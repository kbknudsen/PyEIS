<p align="center">
  <img src="https://github.com/kbknudsen/PyEIS/blob/master/pyEIS_images/PyEIS_logo.png">
</p>

This repository contains PyEIS, A Python-based Electrochemical Impedance Spectroscopy analyzer and simulator. The software is designed to perform impedance simulations and analyze experimental data through the application of circuit elements. Physical processes in electrochemical systems can be represented by analog circuits containing capacitors (C), resistors (R), inductors (I), and some distributed elements such as constant-phase- (Q) and Warburg elements (W). These features make it possible to understand kinetics, double-layers, and mass-transport for a large range of electrochemical applications. 

PyEIS has nine main features:
- Currently contains 26 built-in equivalent circuits 
- Automated graphical representation in Nyquist and Bode plots with a number of plotting options
- Capable of importing experimental data from Bio-Logic's EC-Lab '.mpt', Gamry's '.DTA', and Solartron's '.z' files
- Experimental data validation and quality assessment through Boukamp's linear Kramers-Kronig analysis [1] with an automated optimization function that ensures an optimal number of -(RC)- elements ensuring data is neither over- or under-fitted [2]
- Ability to fit experimental data through the weighed complex non-linear least squares fitting procedure using the lmfit package [3] with any built-in equivalent circuit
- Batch fitting capabilities that do not require any additional key strokes
- Extraction of fitted parameters for fast post-analysis
- Open-source platform that makes it feasible to include any new equivalent circuit
- Tutorials for simulating impedance, importing experimental data, and fitting experimental data


Following the license agreement, please use the following citation: [![DOI](https://zenodo.org/badge/159585045.svg)](https://zenodo.org/badge/latestdoi/159585045)


Author: Kristian B. Knudsen (kknu@berkeley.edu || kristianbknudsen@gmail.com)

## How to use PyEIS
PyEIS works in a python 3 environment. It was built, tested, and automated in Jupyter lab and Spyder. To use PyEIS, an independent interface is not available as impedance fitting and post analysis of fitted parameters would become a two-step process. Instead PyEIS works directly in a Python interface and fitted parameters are automatically output in variables directly accessible to plot or analyze vs. potential, current, state-of-charge, cycle number, time, etc. allowing for fast analysis.


The following command overview and two notebooks are tutorials that in a step-by-step manner introduce the functionality of PyEIS:

- [PyEIS command overview](https://github.com/kbknudsen/PyEIS/blob/master/Tutorials/PyEIS_command_overview.pdf)
- [Simulations with PyEIS](https://github.com/kbknudsen/PyEIS/blob/master/Tutorials/PyEIS_simulation_tutorial.ipynb)
- [Experimental Data Extraction and Fitting with PyEIS](https://github.com/kbknudsen/PyEIS/blob/master/Tutorials/PyEIS_experimental-data_tutorial.ipynb)

The [PyEIS command overview](https://github.com/kbknudsen/PyEIS/blob/master/Tutorials/PyEIS_command_overview.pdf) gives a brief overview of the main functionalities and their dependents. The [Simulations with PyEIS](https://github.com/kbknudsen/PyEIS/blob/master/Tutorials/PyEIS_simulation_tutorial.ipynb) notebook covers simulating and plotting impedance spectra’s with different built-in equivalent circuits, fitting generated data with equivalent circuits, and extracting fitted parameters. The [Experimental Data Extraction and Fitting with PyEIS](https://github.com/kbknudsen/PyEIS/blob/master/Tutorials/PyEIS_experimental-data_tutorial.ipynb) notebook covers how to import experimental data, perform linear Kramers-Kronig analysis of the experimental data to assess data quality, how to mask data, fitting and plotting experimental data using equivalent circuits, assessing quality of fit using relative residuals, and extracting fitted parameters such as resistors and capacitors for further post-analysis.

The built-in equivalent circuits are illustrated in the following figure. Here Boukamp's simple notation of circuits [5] is used in the "trivial term", while "Simulation function" describes the function that needs to be called to perform simulations, and "Fit string" describes a circuit string that needs be called in the fitting function.

<p align="center">
  <img src="https://github.com/kbknudsen/PyEIS/blob/master/pyEIS_images/Equivalent_Circuits_avaliable.png">
</p>

## Software and Installation
PyEIS is available on PyPI and can be install using the following command

> - pip install PyEIS==1.0.10

### Dependencies
The following libraries are required by PyEIS

- numpy >= 1.13.3
- scipy >= 1.0.1
- pandas >= 0.22.0
- mpmath >= 1.1.0
- lmfit >= 0.9.7
- matplotlib >= 2.2.2
- seaborn >= 0.8.1


### [Release history](https://github.com/kbknudsen/PyEIS/blob/master/Changes.txt)

## License
All files in this repository including code, readme, logos, and figures are released under the Apache 2.0 License. Learn more at the [Open Source Initiative](https://opensource.org/licenses/Apache-2.0).

## Acknowledgements
PyEIS is the accumulation of Kristian's work studying kinetics, double-layer and capacitive effects, and mass transport limitations in electrochemical cells with Electrochemical Impedance Spectroscopy during his PhD at The Technical University of Denmark, Department of Energy under the supervision of Ass. Prof. Johan Hjelm. He currently maintains a Post Doctoral position at the University of California, Berkeley at the Department of Chemical Engineering with Ass. Prof. Bryan D. McCloskey.

Funding is acknowledged from NASA ARMD Convergent Aeronautics Solutions (CAS) Project (Cooperative Agreement NNX16AR82A).

[Google scholar site](https://scholar.google.dk/citations?user=gTBdd5wAAAAJ&hl=da).

## References
[1] Boukamp B.A., Solid State Ionics 20, 31-44 (1986), "A Linear Kronig-Kramers Transform Test for Immittance Data Validation"

[2] M. Schönleber, D. Klotz, and E. Ivers-Tiffée, Electrochimica Acta, 131, 20–27 (2014).

[3] Newville M., et al. "LMFIT: Non-Linear Least-Square Minimization and Curve-Fitting for Python" (2014) https://doi.org/10.5281/zenodo.11813

[4] Oliphant T.E (2006) "A guide to NumPy" Trelgol Publishing

[5] Boukamp B.A., "Equivalent Circuit. User Manual" University of Twente, The Netherlands, 1989, 2nd edn.

[6] Johansson F. (2013) "Mpmath: A Python Library for arbitrary-precision floating-point arithmetric" v. 0.18, http://mpmath.org

[7] McKinney W., Proceedings of the 9th Python in Science Conference, 51-56 (2010) "Data Structures for Statistical Computing in Python"

[8] Hunter, J.D. Computing in Science & Engineering, 9, 90-95 (2007) "Matplotlib: A 2D Graphics Environment", DOI:10.1109/MCSE.2007.55