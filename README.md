# QD_laser_models

## Description

This repository contains the Python class QD that simulates models of quantum dot lasers that include coherent fields. These are introduced in a series of papers:

[1] **Coherent-Incoherent Model (CIM)** - Carroll, M. A.;
D'Alessandro, G.; Lippi, G. L.; Oppo, G.-L. & Papoff, F. “Thermal,
Quantum Antibunching and Lasing Thresholds from Single Emitters to
Macroscopic Devices” Phys. Rev. Lett., 2021, **126**, 063902 - DOI:
[10.1103/PhysRevLett.126.063902](https://doi.org/10.1103/PhysRevLett.126.063902)_

[2] **Two-Particle Model (TPM)** - Papoff, F.; Carroll, M. A.; Lippi,
G. L.; Oppo, G.-L. & D’Alessandro, G. “Quantum correlations, mixed
states, and bistability at the onset of lasing” Physical Review A,
2025, **111**, l011501 - DOI: [10.1103/PhysRevA.111.L011501]
(https://doi.org/10.1103/PhysRevA.111.L011501)

[3] **Two-Particle Model with negligible fermion-fermion interactions
(TPM\_1F) and models with different quantum dots (CIM\_d, TPM\_d and
TPM_1F\_d)** - D’Alessandro, G.; Lippi, G. L. & Papoff, F. “Threshold
Behavior in Quantum Dot Nanolasers: Effects of Inhomogeneous
Broadening” - Submitted to Phys. Rev. A

Please cite these papers if you use these models.

Contact: Francesco Papoff [f.papoff@strath.ac.uk]

## Documentation

Please look at the “Quantum dots with coherent fields” site on [Read the Docs](https://app.readthedocs.org/):  

[https://qd-laser-models.readthedocs.io/en/latest](https://qd-laser-models.readthedocs.io/en/latest/)

## Files

The quantum dot class file, **QD.py**, is the only file required to run the various quantum dots models. It can be downloaded and installed indepedently of any other file in this repository. See the file header for the Python modules required by it.  

The **Makefile** and the folders **docs** and **builds** are required to produce the documentation using [sphinx](https://www.sphinx-doc.org/en/master/index.html). You can type `make html` to produce the html version of the documentation (this requires the Python modules sphinx and sphinx-rtd-theme to be installed).  

The files **.readthedocs.yaml** and **docs/requirements.txt** are required by  [Read the Docs](https://app.readthedocs.org/) to produce the documentation.  

## License

[CC BY 4.0](https://creativecommons.org/licenses/by/4.0/)

## Disclaimer

This software is provided "as is," without warranty of any kind, express or 
implied, including but not limited to the warranties of merchantability, 
fitness for a particular purpose, and non-infringement. In no event shall 
the authors or copyright holders be liable for any claim, damages, or 
other liability, whether in an action of contract, tort, or otherwise, 
arising from, out of, or in connection with the software or the use or 
other dealings in the software.

The user assumes all responsibility and risk for the use of this software. 
We make no representations or warranties about the suitability, reliability, 
availability, timeliness, and accuracy of the software for any purpose.
