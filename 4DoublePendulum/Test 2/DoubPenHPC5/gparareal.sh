#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --mem-per-cpu=3700
#SBATCH --time=48:00:00

### INCREASE --cpus-per-task above when out-of-memory occurs 
### in the management process. Next, pool of workers created in the submitted
### MATLAB script below will have a separate memory allocation controlled by
### the following MATLAB_* variables, and, as usual, a larger number of CPUs 
### (workers) allow for a larger allocation

module purge
module load MATLAB/2022a
module load iccifort/2019.5.281
module load MATLAB-parenv/0.0-iccifort-2019.5.281

# set the number of nodes to run MATLAB on
export MATLAB_NNODES=16

# number of tasks per node, giving NNODES*NTASKS_PER_NODE workers
export MATLAB_NTASKS_PER_NODE=32

# each worker can use multiple threads, check if it gives a better performance
export MATLAB_CPUS_PER_TASK=1

# run the code
matlab -batch "pool"

# launch command for older MATLAB versions
# matlab -nodisplay -nosplash < pool.m
