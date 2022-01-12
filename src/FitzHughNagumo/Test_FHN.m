%% FitzHugh-Nagumo model
%Here we solve the FHN system - a pair of coupled stiff ODEs. To run the 
% code, run section 1 below.

%WARNING: 
% The code here is designed to run on a HPC system with at least 
% 40 compute cores and high RAM. Reduce the value of Nf in section 1
% to enable running on a laptop - noting that results will be different
% to the manuscript results and parallel speedup may not be realised.

%% SECTION 1: Solve system
% This section solves the system using GParareal and parareal, comparing
% against the fine solver solution.

clear; close all; clc;

%initialise the FHN model in a function handle
a = 0.2;
b = 0.2;
c = 3;
f = @(t,u)([c*(u(1) - ((u(1)^3)/3) + u(2)); -(1/c)*(u(1) - a + b*u(2))]);

%other inputs to GParareal (and parareal)
tspan = [0,40];                    %time interval of integration
u0 = [-1,1];                       %initial condition for ODE
epsilon = 10^(-6);                 %error tolerance
N = 40;                            %no. of time sub-intervals (processors)
Ng = 160;                          %coarse time steps
Nf = 160000;                       %fine time steps
G = 'RK2';                         %coarse solver (see RK.m file)
F = 'RK4';                         %fine solver   (see RK.m file)

% Gaussian Process emulator inputs (uses GPstuff package)
K = cell(2,1);                                          %cell array stores covariance kernel structure for each output dimension
K{1} = gpcf_sexp('lengthScale',1, 'magnSigma2',1);      %square exponential kernel (isotropic)
K{2} = gpcf_sexp('lengthScale',1, 'magnSigma2',1);      %square exponential kernel (isotropic)
like = lik_gaussian();                                  %Gaussian likelihood structure

%solve FHN model using GParareal and parareal
[t,u,err,k,gp,xx,DD] = GParareal(f,tspan,u0,N,Ng,Nf,epsilon,F,G,K,like);
[~,u_p,err_para,k_para] = parareal(f,tspan,u0,N,Ng,Nf,epsilon,F,G);

% display optimised GP hyperparameters from GParareal (note they are in log
% form)
fprintf('Optimised GP hyperparameters - output one: \n')
[w1,s1] = gp_pak(gp{1});  disp(s1), disp(w1)
fprintf('Optimised GP hyperparameters - output two: \n')
[w2,s2] = gp_pak(gp{2});  disp(s2), disp(w2);

%run fine solver
n = length(u0);                         % no. of ODE output dimensions
dt = (tspan(2)-tspan(1))/Nf;            % size of fine time step
t_fine = (tspan(1):dt:tspan(end));      % fine time step mesh
[~,F_sol] = RK(t_fine,u0,f,F);          % fine solution


%PLOTS
% Fig 1: GParareal/parareal solutions vs. fine solution

%integrate the ODE using initial values found by GParareal and parareal
fine_trajecs1 = cell(N,1);
fine_trajecs2 = cell(N,1);
parfor i = 1:N
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
u_GPara = vertcat(fine_trajecs1{:,1});            % probabilistic sol
u_para = vertcat(fine_trajecs2{:,1});            % parareal sol


% plot the solutions against time
figure(1)

subplot(2,1,1)
hold on
plot(t_fine,F_sol(:,1),'k')
plot(t_fine((1:(Nf/N):end)),u_GPara((1:(Nf/N):end),1),'ob')
plot(t_fine((1:(Nf/N):end)),u_para((1:(Nf/N):end),1),'*r')
hold off
xlim(tspan); ylim([-3 3])
set(gca,'xticklabels',[])
ylabel('$u_1(t)$','interpreter','latex');
box on; grid on;

subplot(2,1,2)
hold on
plot(t_fine,F_sol(:,2),'k')
plot(t_fine((1:(Nf/N):end)),u_GPara((1:(Nf/N):end),2),'ob')
plot(t_fine((1:(Nf/N):end)),u_para((1:(Nf/N):end),2),'*r')
hold off
xlim(tspan); ylim([-1.5 1.5])
xlabel('$t$','interpreter','latex'); ylabel('$u_2(t)$','interpreter','latex');
box on; grid on;

legend({'Fine','GParareal','Parareal'},'Interpreter','latex','numcolumns',3,'location','southwest')
hold off


%Fig 2: Convergence plot (GParareal vs. parareal)

%calculate maximum error during each iteration (for each algorithm)
err(cumsum(cumsum(err~=0)) == 1) = 0; err(end,end) = eps; errs = max(err,[],1); 
err_para(cumsum(cumsum(err_para~=0)) == 1) = 0; err_para(end,end) = eps; errs_para = max(err_para,[],1); 

figure(2)
hold on
plot((1:k_para),errs_para(1:k_para),'-or','LineWidth',1.2);
plot((1:k),errs(1:k),'-ob','LineWidth',1.2);
yline(epsilon,'--k','LineWidth',1.2)
xlabel('$k$','Interpreter','latex'); ylabel('Max. absolute error','Interpreter','latex');
set(gca,'yscale','log')
grid on; box on;
xlim([1,max([k,k_para])]); ylim([10^(-8),10^(1)]);
xticks((1:max([k,k_para]))); %yticks((0:0.5:5));
legend('northeast',{'Parareal','GParareal','Tolerance'})
hold off

%Fig 3: Error plot (GParareal/parareal vs. fine solution) against time

%calculate absolute error at each time step (for each algorithm)
prob_error1 = abs(F_sol(:,1) - u_GPara(:,1)); prob_error1(prob_error1==0) = 10^(-40);
para_error1 = abs(F_sol(:,1) - u_para(:,1)); para_error1(para_error1==0) = 10^(-40);

prob_error2 = abs(F_sol(:,2) - u_GPara(:,2)); prob_error2(prob_error2==0) = 10^(-40);
para_error2 = abs(F_sol(:,2) - u_para(:,2)); para_error2(para_error2==0) = 10^(-40);

figure(3)
subplot(2,1,1)
hold on
plot(t_fine,para_error1,'r','LineWidth',1.2)
plot(t_fine,prob_error1,'b','LineWidth',1.2)
hold off
set(gca,'yscale','log')
yticks(10.^[-15,-10,-5,0])
xlim(tspan)
ylim([10^(-15) 10^(0)]);
set(gca,'xticklabels',[])
ylabel('$u_1$ error','interpreter','latex');
box on; grid on;
legend({'Fine - Parareal','Fine - GParareal'},'interpreter','latex','location','northwest','numcolumns',2)

subplot(2,1,2)
hold on
plot(t_fine,para_error2,'r','LineWidth',1.2)
plot(t_fine,prob_error2,'b','LineWidth',1.2)
hold off
set(gca,'yscale','log')
yticks(10.^[-15,-10,-5,0])
xlim(tspan)
ylim([10^(-15) 10^(0)]);
xlabel('$t$','interpreter','latex'); ylabel('$u_2$ error','interpreter','latex');
box on; grid on;
