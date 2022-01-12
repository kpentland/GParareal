# GParareal: A time-parallel ODE solver using Gaussian process emulation

This repository contains sample code for the pre-print by Pentland, Tamborrino, Sullivan, Buchanan, and Appel - "GParareal: A time-parallel ODE solver using Gaussian process emulation" (insert arXiv link here).

The code is written in MATLAB and requires the use of the following toolboxes (note there may be dependencies I have overlooked):
* Parallel Computing Toolbox.
* GPstuff (download/install instructions at: https://github.com/gpstuff-dev/gpstuff).

## Files

* *GParareal.m*: function file containining the GParareal algorithm. 
* *parareal.m*: function file containining the parareal algorithm.
* *RK.m*: function file containining 1st, 2nd, 3rd, and 4th order Runge-Kutta methods.
* *FitzHughNagumo*: folder containing a test run of the three functions above on the FitzHugh-Nagumo model (open the 'Test_FHN.m script). 
* *Rossler*: folder containing a test run of the three functions above on the Rossler system (open the 'Test_rossler.m script).

## Authors

* Kamran Pentland - Mathematics Institute, University of Warwick
* Massimiliano Tamborrino - Dept. of Statistics, University of Warwick
* Tim Sullivan - Mathematics Institute, University of Warwick
* James Buchanan - Cuham Centre for Fusion Energy, Abingdon, Oxfordshire
* Lynton Appel - Cuham Centre for Fusion Energy, Abingdon, Oxfordshire
