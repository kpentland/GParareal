%% Arenstorf orbits (2D 2nd order to 4D 1st order)
%Here we solve the Arenstorf orbit equations. Plots are described below.  
%To test, run the following script in its entirety. 

clear; close all; clc;

% INPUTS
%function handle for ODE
mu1 = 0.012277471;
mu2 = 1 - mu1;
f = @(t,u)([u(2);u(1)+2*u(4) - mu2*((u(1)+mu1)/(((u(1)+mu1)^2 + u(3)^2)^(1.5)))...
    - mu1*((u(1)-mu2)/(((u(1)-mu2)^2 + u(3)^2)^(1.5)));u(4);u(3) - 2*u(2)...
    - mu2*(u(3)/(((u(1)+mu1)^2 + u(3)^2)^(1.5))) - mu1*(u(3)/(((u(1)-mu2)^2 ...
    + u(3)^2)^(1.5)))]);    
auto = 1;                              %is the system autonomous (yes=1, no=0)
%tspan = [0,17.06521656015796]          %time interval
tspan = [0,34];                        %time interval
u0 = [0.994,0,0,-2.00158510637908];    %intial conditions
N = 250;                                %no. of time sub-intervals steps
Ng = 250;                              %no. of coarse steps
Nf = 80000;                            %no. of fine steps
epsilon = 10^(-8);                     %error tolerance 
G = 'RK2';                             %coarse solver
F = 'RK4';                             %fine solver

% Gaussian Process prior inputs
K = cell(4,1);                                          %covariance kernel structure required for each output
K{1} = gpcf_sexp('lengthScale',1, 'magnSigma2',1);      %square exponential kernel (isotropic)
K{2} = gpcf_sexp('lengthScale',1, 'magnSigma2',1);      %square exponential kernel (isotropic)
K{3} = gpcf_sexp('lengthScale',1, 'magnSigma2',1);      %square exponential kernel (isotropic)
K{4} = gpcf_sexp('lengthScale',1, 'magnSigma2',1);      %square exponential kernel (isotropic)
like = lik_gaussian();                                  %Gaussian likelihood with prior scale parameter

%fine time step and mesh (n = no. of dimensions)
n = length(u0);
dt = (tspan(2)-tspan(1))/Nf;
t_fine = (tspan(1):dt:tspan(end));


%solve problem with original IC
[t,u,err,k,gp,xx,DD] = PPODE(f,auto,tspan,u0,N,Ng,Nf,epsilon,F,G,K,like);     % PPODE solution
[~,F_sol] = RK(t_fine,u0,f,F);                                           % fine solution
[~,u_p,err_para,k_para] = parareal(f,tspan,u0,N,Ng,Nf,epsilon,F,G);      % parareal solution


% display optimised GP hyperparameters
fprintf('Optimised GP hyperparameters - output one: \n')
[w1,s1] = gp_pak(gp{1});  disp(s1), disp(w1)
fprintf('Optimised GP hyperparameters - output two: \n')
[w2,s2] = gp_pak(gp{2});  disp(s2), disp(w2)
fprintf('Optimised GP hyperparameters - output three: \n')
[w3,s3] = gp_pak(gp{3}); disp(s3), disp(w3)
fprintf('Optimised GP hyperparameters - output four: \n')
[w4,s4] = gp_pak(gp{4}); disp(s4), disp(w4)

%%PLOTS
% Fig 1: PPODE/parareal solutions vs. fine

%integrate the ODE using initial values found by PPODE and parareal
fine_trajecs1 = cell(N,1);
fine_trajecs2 = cell(N,1);
for i = 1:N
    [~,temp1] = RK((t(i):dt:t(i+1)),u(i,(n*(k-1)+1:n*k)),f,F);
    [~,temp2] = RK((t(i):dt:t(i+1)),u_p(i,(n*(k_para-1)+1:n*k_para)),f,F);
    if i < N
    fine_trajecs1{i,1} = temp1(1:end-1,:);                 
    fine_trajecs2{i,1} = temp2(1:end-1,:);                 
    else
    fine_trajecs1{i,1} = temp1;                 
    fine_trajecs2{i,1} = temp2;                 
    end
end
u_prob = vertcat(fine_trajecs1{:,1});            % probabilistic sol
u_para = vertcat(fine_trajecs2{:,1});            % parareal sol


% plot the solutions
figure(1)
hold on
plot(F_sol(:,1),F_sol(:,3),'k')
plot(u_para((1:(Nf/N):end),1),u_para((1:(Nf/N):end),3),'*r')
plot(u_prob((1:(Nf/N):end),1),u_prob((1:(Nf/N):end),3),'ob')
xlabel('$x$','Interpreter','latex'); ylabel('$y$','Interpreter','latex');
grid on; box on;
%axis([0 4 0 5])
%xticks((0:0.5:4)); yticks((0:0.5:5));
legend({'Fine','Parareal','PPODE'},'Interpreter','latex','location','northeast')
hold off



%Fig 2: Convergence plot (PPODE vs. parareal)

%calculate maximum error during each iteration (for each algorithm)
err(cumsum(cumsum(err~=0)) == 1) = 0; err(end,end) = eps; errs = max(err,[],1); 
err_para(cumsum(cumsum(err_para~=0)) == 1) = 0; err_para(end,end) = eps; errs_para = max(err_para,[],1); 

figure(2)
hold on
plot((1:k_para),errs_para(1:k_para),'-ok','LineWidth',1.2);
plot((1:k),errs(1:k),'-ob','LineWidth',1.2);
yline(epsilon,'--k')
xlabel('$k$ (Iterations)','Interpreter','latex'); ylabel('Max. Absolute Error','Interpreter','latex');
set(gca,'yscale','log')
grid on; box on;
xlim([1,max([k,k_para])]); ylim([10^(-10),10^(2)]);
xticks((1:max([k,k_para]))); %yticks((0:0.5:5));
legend('northeast',{'Parareal','PPODE','Tolerance'})
hold off

%Fig 3: Error plot (PPODE/parareal vs. fine solution)

%calculate absolute error at each time step (for each algorithm)
prob_error1 = abs(F_sol(:,1) - u_prob(:,1)); prob_error1(prob_error1==0) = 10^(-40);
para_error1 = abs(F_sol(:,1) - u_para(:,1)); para_error1(para_error1==0) = 10^(-40);

prob_error2 = abs(F_sol(:,3) - u_prob(:,3)); prob_error2(prob_error2==0) = 10^(-40);
para_error2 = abs(F_sol(:,3) - u_para(:,3)); para_error2(para_error2==0) = 10^(-40);

figure(3)
subplot(2,1,1)
hold on
plot(t_fine,para_error1,'r','LineWidth',1.2)
plot(t_fine,prob_error1,'b','LineWidth',1.2)
hold off
set(gca,'yscale','log')
xlim(tspan)
ylim([10^(-15) 10^(0)]);
xlabel('$t$','interpreter','latex'); ylabel('$x$ error','interpreter','latex');
box on; grid on;
legend({'Fine - Parareal','Fine - PPODE'},'interpreter','latex','location','northwest')

subplot(2,1,2)
hold on
plot(t_fine,para_error2,'r','LineWidth',1.2)
plot(t_fine,prob_error2,'b','LineWidth',1.2)
hold off
set(gca,'yscale','log')
xlim(tspan)
ylim([10^(-15) 10^(0)]);
xlabel('$t$','interpreter','latex'); ylabel('$y$ error','interpreter','latex');
box on; grid on;



%% Solve using prior data
%Here we solve the Arenstorf orbit equations. Plots are described below.  
%To test, run the following script in its entirety. 

clear; close all; clc;

% INPUTS
%function handle for ODE
mu1 = 0.012277471;
mu2 = 1 - mu1;
f = @(t,u)([u(2);u(1)+2*u(4) - mu2*((u(1)+mu1)/(((u(1)+mu1)^2 + u(3)^2)^(1.5)))...
    - mu1*((u(1)-mu2)/(((u(1)-mu2)^2 + u(3)^2)^(1.5)));u(4);u(3) - 2*u(2)...
    - mu2*(u(3)/(((u(1)+mu1)^2 + u(3)^2)^(1.5))) - mu1*(u(3)/(((u(1)-mu2)^2 ...
    + u(3)^2)^(1.5)))]);    
auto = 1;                              %is the system autonomous (yes=1, no=0)
tspan = [0,17.06521656015796];         %time interval
u0 = [0.994,0,0,-2.00158510637908];    %intial conditions
N = 40;                               %no. of time sub-intervals steps
Ng = 400;                              %no. of coarse steps
Nf = 80000;                            %no. of fine steps
epsilon = 10^(-8);                     %error tolerance 
G = 'RK2';                             %coarse solver
F = 'RK4';                             %fine solver

% Gaussian Process prior inputs
K = gpcf_sexp('lengthScale',1, 'magnSigma2',1);      %square exponential kernel (isotropic)
like = lik_gaussian();                               %Gaussian likelihood with prior scale parameter

%fine time step and mesh (n = no. of dimensions)
n = length(u0);
dt = (tspan(2)-tspan(1))/Nf;
t_fine = (tspan(1):dt:tspan(end));


%run F and G over selected ICs in the domain domain
ns = 400;
xs1 = linspace(-2,2,sqrt(ns));
xs2 = linspace(-1,1,sqrt(ns));
[xs1,xs2] = meshgrid(xs1,xs2); 
xx = repmat([xs1(:), xs2(:)],2,1);

F_data = zeros(length(xx),2);
G_data = zeros(length(xx),2);
for i = 1:length(xx)
    [~,F_sol] = RK((tspan(1):dt:(tspan(2)/N)),xx(i,:),f,F);
    [~,G_sol] = RK((tspan(1):((tspan(2)-tspan(1))/Ng):(tspan(2)/N)),xx(i,:),f,G);
    F_data(i,:) = F_sol(end,:);
    G_data(i,:) = G_sol(end,:);
end
DD = F_data - G_data;
xx(abs(min(DD,[],2))>10,:) = [];
DD(abs(min(DD,[],2))>10,:) = [];

% %solve problem with original IC
% [t,u,u_var,err,k,gp,~,~] = PPODEv2(f,tspan,u0,N,Ng,Nf,epsilon,F,G,K,like); % prob solver
% [~,F_sol] = RK(t_fine,u0,f,F);                                                 % fine solver
% [~,u_p,err_para,k_para] = parareal(f,tspan,u0,N,Ng,Nf,epsilon,F,G);            % parareal

%solve with alternate IC (using perviously obtained data and without)
u0_new = [-1,1];

%solve problem using without prior data
[t,u2,u_var2,err2,k2,~,~,~] = PPODEv2(f,tspan,u0_new,N,Ng,Nf,epsilon,F,G,K,like);       % prob solver
%solve problem using previously obtained data
[~,u3,u_var3,err3,k3,gp,~,~] = PPODEv2(f,tspan,u0_new,N,Ng,Nf,epsilon,F,G,K,like,xx,DD); % prob solver
%solve using fine solver and parareal
[~,F_sol2] = RK(t_fine,u0_new,f,F);                                                            % fine solver
[~,u_p2,err_para2,k_para2] = parareal(f,tspan,u0_new,N,Ng,Nf,epsilon,F,G);                     % parareal
 


% use the ICs found by each algorithm to integrate in parallel
fine_trajecs1 = cell(N,1);
fine_trajecs2 = cell(N,1);
fine_trajecs3 = cell(N,1);
fine_trajecs4 = cell(N,1);
fine_trajecs5 = cell(N,1);
for i = 1:N
%     [~,temp1] = RK((t(i):dt:t(i+1)),u(i,(n*(k-1)+1:n*k)),f,F);
%     [~,temp2] = RK((t(i):dt:t(i+1)),u_p(i,(n*(k_para-1)+1:n*k_para)),f,F);
    [~,temp3] = RK((t(i):dt:t(i+1)),u2(i,(n*(k2-1)+1:n*k2)),f,F);
    [~,temp4] = RK((t(i):dt:t(i+1)),u3(i,(n*(k3-1)+1:n*k3)),f,F);
    [~,temp5] = RK((t(i):dt:t(i+1)),u_p2(i,(n*(k_para2-1)+1:n*k_para2)),f,F);
    if i < N
%         fine_trajecs1{i,1} = temp1(1:end-1,:);
%         fine_trajecs2{i,1} = temp2(1:end-1,:);
        fine_trajecs3{i,1} = temp3(1:end-1,:);
        fine_trajecs4{i,1} = temp4(1:end-1,:);
        fine_trajecs5{i,1} = temp5(1:end-1,:);
    else
%         fine_trajecs1{i,1} = temp1;
%         fine_trajecs2{i,1} = temp2;
        fine_trajecs3{i,1} = temp3;
        fine_trajecs4{i,1} = temp4;
        fine_trajecs5{i,1} = temp5;
    end
end
% u_prob = vertcat(fine_trajecs1{:,1});             % prob solution (old IC)
% u_para = vertcat(fine_trajecs2{:,1});             % parareal sol (old IC)
u_prob2 = vertcat(fine_trajecs3{:,1});            % prob solution (new IC - without data)
u_prob3 = vertcat(fine_trajecs4{:,1});            % prob solution (new IC - with data)
u_para2 = vertcat(fine_trajecs5{:,1});            % parareal sol (new IC)
