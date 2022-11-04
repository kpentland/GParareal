# GParareal: A time-parallel ODE solver using Gaussian process emulation

This repository contains sample code for the pre-print by Pentland, Tamborrino, Sullivan, Buchanan, and Appel - "GParareal: A time-parallel ODE solver using Gaussian process emulation" (https://arxiv.org/abs/2201.13418). Using the 'TESTNAME_Results.m' in each folder, one should be able to re-create the results from the paper.

The code is written in MATLAB and requires the use of the following toolboxes (note there may be dependencies I have overlooked):
* Parallel Computing Toolbox.


## Files

* *1FHN*: contains experiments on the FitzHugh-Nagumo system (open the 'FHN_Results.m script). 
* *2Rossler*: contains experiments on the Rossler system (open the 'Rossler_Results.m script). 
* *3Nonautonomous*: contains experiments on a nonautonomous ODE system (open the 'NonAuto_Results.m script). 
* *4DoublePendulum*: contains experiments on the classic double pendulum system (open the 'DoublePen_Results.m script). 

## Authors

* Kamran Pentland - Mathematics Institute, University of Warwick
* Massimiliano Tamborrino - Dept. of Statistics, University of Warwick
* Tim Sullivan - Mathematics Institute, University of Warwick
* James Buchanan - Cuham Centre for Fusion Energy, Abingdon, Oxfordshire
* Lynton Appel - Cuham Centre for Fusion Energy, Abingdon, Oxfordshire
