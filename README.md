# Time-parallelisation via GP emulation

The code is written in MATLAB and requires the use of the following toolboxes (noting there may be dependencies I have overlooked):
* Parallel Computing Toolbox.
* GPstuff package (download/install instructions at: https://github.com/gpstuff-dev/gpstuff)

Note that I've switched off the parallel components, i.e. all 'parfor' loops are serial 'for' loops, as we obtain no speed when solving these small test systems here. 

## Files

*src*: contains source code for the project.

* Parareal: function for solving systems of ODEs using time-parallelisation.
* PPODE:    function for solving systems of ODEs using time-parallelisation and GP emulation.
* RK:       function for solving systems of ODEs (explicit RK1/2/3/4 options available)
* Testbed:  script containing example problems solved using the previous three functions. 
