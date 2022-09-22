function [t,u,err,k,gp_kers,xx,DD,T] = GParareal(f,tspan,u0,N,Ng,Nf,epsilon,F,G,kernel,varargin)
% Code to solve a system of (autonomous) ODEs in parallel (using a Gaussian
% process emulator). We assume no covariance between outputs dimensions of 
% the GP.

%Inputs:
% f:           Function handle for function to be solved (i.e. f = @(t,u)([u(1);u(2)])
% tspan:       Time interval over which to integrate (i.e. [0,12])
% u0:          Initial conditions at tspan(1) (i.e. [0,1])
% N:           Number of 'proccesors' (temporal sub-intervals) (i.e. N = 40)
% Ng:          Number of coarse time steps (i.e. Ng = 40)
% Nf:          Number of fine times steps (i.e. Nf = 4000)
% epsilon:     Error tolerance (i.e. 10^(-6))
% F:           Selected fine solver (i.e. 'RK4') 
% G:           Selected coarse solver (i.e. 'RK1') 
% kernel:      Structure containing covariance function and hyperparameters for GP (i.e. struct('func',@isoSE_cov,'theta',[1,1],'jitter',10*eps))
% varargin     Variable inputs x and D can either be both omitted or both be present.
% x:           Vector of legacy initial value inputs - used to train GP
% D:           Vector of legacy output data on (F-G)(x) - used to train GP

%Outputs:
% t:           Vector of time sub-intervals (at which solutions located)
% u:           Solution to ODE system on the mesh given by 't'
% err:         Successive errors at each time sub-interval and each k
% k:           Iterations taken until convergence
% gp_kers:     Structure containing the final GP kernel information
% xx:          Accumulated vector of initial value inputs used to train GP (acquisition data)
% DD:          Accumulated vector of data on (F-G)(xx) used to train GP (acquisition data)
% T:           Cell array of timings for G and F solvers, as well as the GP ([tG,tF,tGP])

% INITIALISATION
n = length(u0);                    %dimension of the ODE system
L = tspan(2) - tspan(1);           %length of interval
L_sub = L/N;                       %length of sub-intervals
dT = L/Ng;                         %coarse time step size
dt = L/Nf;                         %fine time step size
t = (tspan(1):L_sub:tspan(2));     %time sub-intervals (the mesh)  
t_shift = t(2:end);                %shifted mesh for parfor loops below
I = 1;                             %counter for how many intervals have converged

% error catch: sub-interval, coarse, and fine time steps must be multiples of each other
if mod(Ng,N)~=0 || mod(Nf,Ng)~=0
    fprintf("Nf must be a multiple of Ng and Ng must be a multiple of N - change time steps!")
    return
end

% solution storage matrices (sub-interval mesh x system dimension*iterations)
u = NaN(N+1,n*(N+1));        %predictor-corrected (refined) solutions
uG = NaN(N+1,n*(N+1));       %coarse solutions
uF = NaN(N+1,n*(N+1));       %fine solutions
err = NaN(N+1,N);            %successive errors

% timing storage matrices and unconverged time slice counter
tG = NaN(N+1,N+1);
tF = NaN(N+1,N+1);
tGP = NaN(2,N);
slices = NaN(1,N);

%initialuse data storage variables
if isempty(varargin)
    x = [];                      %empty array to store acquisition initial values (for GP)
    D = [];                      %empty array to store corresponding acquisition propagated intial values (for GP)
elseif length(varargin) == 2
    x = varargin{1};             %use legacy data if available
    D = varargin{2};
else
    error('Check GParareal input parameters are correct.')
end

% pre-set the exact initial condition at the start of each iteration
u(1,:) = repmat(u0,1,N+1);
uG(1,:) = u(1,:);
uF(1,:) = u(1,:);

%initialise a cell array to store each kernel (i.e. it's hyperparams)
clust = 1;
gp_kers = cell(n*clust,1);
gp_kers(:,1) = {kernel};

hyp = cell(1,N+1);                                             %cell array to store hyperparms in each output at each iteration
hyp{1,1} = ones(n*clust,length(kernel.theta));

% MAIN ALGORITHM

%Iteration k = 0
% Step 1: use G (coarse solver) to find approximate solution values on the mesh
[~,temp] = RK(t(1):dT:t(end),u0,f,G);
uG(:,1:n) = temp(1:round(L_sub/dT):end,:); clear temp;
u(:,1:n) = uG(:,1:n);


%Iteration k > 0
for k = 1:N
    
    % give an indication as to which iteration we're at for the command
    % window
    if k == 1
        fprintf('GParareal iteration number (out of %.0f): 1 ',N)
    elseif k == N
        fprintf('%.0f.',N)
    else
        fprintf('%.0f ',k)
    end
    
    % Step 2: integrate using F (fine solver) in parallel with the current best initial
    % values
    dim_indices = (n*(k-1)+1:n*k);        %current iteration indices
    dim_indices_next = ((n*k)+1:n*(k+1)); %next iteration indices
    parfor i = I:N
        s = tic;
%         [~,temp] = RK((t(i):dt:t_shift(i)),u(i,dim_indices),f,F);
        [~,temp] = RK2([t(i),t_shift(i)],dt,u(i,dim_indices),f,F);
        tF(i,k) = toc(s);
        uF(i+1,dim_indices) = temp(end,:);    %save the solution from final time step
    end
    clear temp
    
    % save values forward (as solution at time I+1 is now converged)
    uG(I+1,(n*k)+1:end) = repmat(uG(I+1,dim_indices),1,N-k+1);
    uF(I+1,(n*k)+1:end) = repmat(uF(I+1,dim_indices),1,N-k+1);
    u(I+1,(n*k)+1:end) = repmat(uF(I+1,dim_indices),1,N-k+1);
    I = I+1;
    
    
    % Step 3: GP prior initialisation (using F and G evaluations)
    
    slices(1,k) = N - I + 2;
    
    % collect initial value data (used to run G and F) from previous
    % iteration
    x_new = u((I-1:length(t)-1),dim_indices);
    x = [x;x_new];
    
    % collect the evaluations of F-G from the corresponding initial value
    % data from previous iteration
    D_new = uF((I:length(t)),dim_indices) - uG((I:length(t)),dim_indices);
    D = [D; D_new];
    
%     if k == 1
%         rng(10) % random seed
%         [idx,C] = kmeans(x,clust);
%     else
%         [~,idx_new] = pdist2(C,x_new,'euclidean','Smallest',1);
%         idx = [idx;idx_new'];
%     end

    if k == 1
        start = zeros(clust,n); start(1,:) = u0;
        temp_idx = [1];
        if clust > 1
           for i = 1:clust-1
               a = (1:length(x)); a(temp_idx) = [];
               dists = pdist2(x(temp_idx(end),:),x(a,:),'euclidean');
               [~,max_idx] = max(dists);
               temp_idx = [temp_idx,max_idx];
           end
        end
        [idx,C] = kmeans(x,clust,'Start',x(temp_idx,:));
    else
        [~,idx_new] = pdist2(C,x_new,'euclidean','Smallest',1);
        idx = [idx;idx_new'];
    end


    %infer the hyperparameters using the data (maximum log likelihood - uses fminsearch function)
    % --> infers hyperparams for each output separately (in parallel) using same inputs
    % --> re-uses hyperparams from previous iteration as new priors
    % --> if hyperparams don't change much between iterations, don't re-optimise
    clust_no = mod((1:n*clust)+(clust-1),clust)+1;
    dim = mod((1:n*clust)+(n-1),n)+1;
    
    s = tic;
    if k == 1 || max(vecnorm(hyp{1,k} - hyp{1,k-1},inf,2)) > 10^(-2)
        temp = zeros(n*clust,length(kernel.theta));
        parfor i = 1:n*clust
            gp_kers{i,1} = GPoptimise(x(idx==clust_no(i),:),D(idx==clust_no(i),dim(i)),gp_kers{i,1});
            temp(i,:) = gp_kers{i,1}.theta;
        end
        hyp{1,k+1} = temp;
    else
        hyp{1,k+1} = hyp{1,k};
    end
    tGP(1,k) = toc(s);
    
    % Step 4: 'predictor-corrector' loop that iteratively combmines the
    % current 'best' initial value using a propagation of G and the
    % expected value of the GP posterior (conditioned on our data).
    for i = I:N
        
        % 'best' initial value where we will evaluate the posterior
        x_star = u(i,dim_indices_next);
        
        % determine which cluster the test input belongs to
        [~,idx_test] = pdist2(C,x_star,'euclidean','Smallest',1);
        
        % error catch
        a = 0;
        if isempty(x_star) || ~isreal(x_star) || ~all(isfinite(x_star))
            a = NaN;
            break
        end
        
        %evaluate the GP posterior at input x_star (does each output
        %seperately)
        s = tic;
        post_mean = zeros(size(x_star));
        for j = 1:n
            [post_mean(1,j),~] = GPpredict(x(idx==idx_test,:),D(idx==idx_test,j),x_star,gp_kers{j,1});
        end
        tGP(2,k) = toc(s);
        
        % run G solver on 'best' initial value
        s = tic;
        [~,temp] = RK(t(i):dT:t(i+1),x_star,f,G);
        tG(i,k) = toc(s);
        uG(i+1,dim_indices_next) = temp(end,:); clear temp;
        
        % combine the G solve and the posterior mean
        u(i+1,dim_indices_next) = post_mean + uG(i+1,dim_indices_next);
        
    end
    
    % error catch
    a = 0;
    if sum(isnan(uG(:,[1,dim_indices_next(end)])),'all') ~= 0
        % error("NaN values in initial coarse solve - increase Ng!")
        a = NaN;
        break
    end
    
    % Step 4: Converence check
    % checks whether difference between solutions at successive iterations
    % are small, if so then that time slice is considered converged.
    err(:,k) = vecnorm( u(:,dim_indices_next) - u(:,dim_indices), inf, 2)';  
    
    II = I;
    for p = II+1:N+1
        if err(p,k) < epsilon
            u(p,(n*(k+1))+1:end) = repmat(u(p,dim_indices_next),1,N-k);
            uG(p,(n*(k+1))+1:end) = repmat(uG(p,dim_indices_next),1,N-k);
            uF(p,(n*k)+1:end) = repmat(uF(p,dim_indices),1,N-k+1);
            I = I + 1;
        else
            break
        end
    end
    
    %break the current iteration if all solutions are converged
    if I == N + 1
        break
    end
end

%output the matrices containing the solutions/errors/timings after 1,2,3...k iterations
u = u(:,(n+1):n*(k+1));
err = err(:,1:k);
T = {median(tG,'all','omitnan'),median(tF,'all','omitnan'),tGP(:,1:k),slices(~isnan(slices)),hyp(1,1:k+1),clust,accumarray(idx,1)};

%also output the acquisition F-G data
xx = x;
DD = D;

% error catch
if isnan(a)
    k = NaN;
end

fprintf('Done. \n')
end
