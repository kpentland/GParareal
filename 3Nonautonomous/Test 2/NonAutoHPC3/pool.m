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

f = @(t,u)([-u(2) + u(1)*(u(3)/500 - u(1)^2 - u(2)^2);u(1) + u(2)*(u(3)/500 - u(1)^2- u(2)^2);1]);  %the ODE to be solved for parareal
tspan = [-20,500];                   %time interval of integration
u0 = [0.1,0.1,tspan(1)];             %initial condition for ODE system + initial time of integration
epsilon = 10^(-6);                   %error tolerance
N = 128;                            %no. of time sub-intervals (processors)
Ng = 2*1024;                           %coarse time steps
Nf = Ng*8500000;                    %fine time steps
G = 'RK1';                           %coarse solver
F = 'RK8';                           %fine solver

% Gaussian Process emulator inputs
kernel = struct('func',@isoSE_cov,'theta',[1,1],'jitter',1e-12);

%solve FHN model using GParareal
s = tic; [t,u,err,k,gp_kers,xx,DD,T] = GParareal(f,tspan,u0,N,Ng,Nf,epsilon,F,G,kernel); t1 = toc(s);

fprintf('GParareal is executed in %8.2f seconds on %8.2f cores. \n', t1, N);

clear c
save('GP_NA_N128.mat')

exit;







