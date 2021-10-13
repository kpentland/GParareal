# Time-parallelisation via GP emulation


This code is written in MATLAB and requires the use of the following toolboxes (noting there may be dependencies I have overlooked):
* Parallel Computing Toolbox.
* Statistics and Machine Learning.
* GPstuff package (download/install instructions at: https://github.com/gpstuff-dev/gpstuff)


## Files

*src*: contains source code for the project.

* Parareal: function for solving systems of ODEs using time-parallelisation.
* PPODE:    function for solving systems of ODEs using time-parallelisation and GP emulation.
* RK:       function for solving systems of ODEs (RK1/2/3/4 options available)
* Testbed:  script containing example problems solved using the previous three functions. 


## Authors

* Kamran Pentland - Mathematics Institute, University of Warwick
* Massimiliano Tamborrino - Dept. of Statistics, University of Warwick
* Tim Sullivan - Mathematics Institute, University of Warwick
* Lynton Appel - Centre for Fusion Energy, Culham
* James Buchanan - Centre for Fusion Energy, Culham
