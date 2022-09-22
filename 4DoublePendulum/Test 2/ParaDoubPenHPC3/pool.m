% add parallel environment/scripts files path
parenv_root = getenv('EBROOTMATLABMINPARENV');
addpath(strcat(parenv_root, '/scripts'));

% caputure the parallelisation parameters from ones set in the batch submission script
thds = str2num(getenv('MATLAB_CPUS_PER_TASK'));
tsks = str2num(getenv('MATLAB_NTASKS_PER_NODE'));
nods = str2num(getenv('MATLAB_NNODES'));

% cast into a string variable
submit_arguments = sprintf('-t 48:00:00 --mem-per-cpu=3700 --nodes=%d --ntasks-per-node=%d --cpus-per-task=%d ', nods, tsks, thds);

% set the cluster profile
c = parallel.cluster.Slurm;
c.NumWorkers = tsks*nods;
c.NumThreads = thds;
c.SubmitArguments = submit_arguments;
c.CommunicatingJobWrapper = strcat(parenv_root, '/scripts/communicatingJobWrapper.sh');

% set and show the MPI library in use (has to be in MVAPICH2 location)
mpiLibConf

% Define the job using batch
parpool(c, c.NumWorkers);



% MY PARALLEL CODE GOES HERE

f = @(t,u)([u(2);...
    (-1/(2 - cos(u(1)-u(3))^2))*((u(2)^2)*cos(u(1)-u(3))*sin(u(1)-u(3)) + (u(4)^2)*sin(u(1)-u(3)) + 2*sin(u(1)) - cos(u(1)-u(3))*sin(u(3)) );...
    u(4);...
    (-1/(2 - cos(u(1)-u(3))^2))*(-2*(u(2)^2)*sin(u(1)-u(3)) - (u(4)^2)*sin(u(1)-u(3))*cos(u(1)-u(3)) - 2*cos(u(1)-u(3))*sin(u(1)) + 2*sin(u(3)) )]);    

tspan = [0,80];                      %time interval
u0 = [2,0,0.5,0];                   %intial conditions (good)
N = 128;                             %no. of time sub-intervals steps
Ng = 3072;                            %no. of coarse steps
Nf = Ng*7000000;                      %no. of fine steps
epsilon = 10^(-6);                    %error tolerance 
G = 'RK1';                            %coarse solver
F = 'RK8';                            %fine solver

% Gaussian Process emulator inputs
kernel = struct('func',@isoSE_cov,'theta',[1,1],'jitter',1e-12);

%solve FHN model using GParareal
s = tic; [t,u_p,err_para,k_para,slices] = parareal(f,tspan,u0,N,Ng,Nf,epsilon,F,G); t1 = toc(s);

fprintf('Parareal is executed in %8.2f seconds on %8.2f cores. \n', t1, N);

clear c
save('P_DP_N128.mat')

exit;

