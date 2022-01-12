%% Rossler system
%Here we solve the Rossler system - three coupled ODEs that exhibit chaotic
% behaviour under the parameters given below. To run the code, run 
% section 1 below.

%WARNING: 
% The code here is designed to run on a HPC system with at least 
% 40 compute cores and high RAM. Reduce the value of Nf in section 1
% to enable running on a laptop - noting that results will be different
% to the manuscript results and parallel speedup may not be realised.


%% SECTION 1: Solve system
% This section solves the system using GParareal and parareal, comparing
% against the fine solver solution.

clear; close all; clc;

% INPUTS 
a = 0.2; b = 0.2; c = 5.7;
f = @(t,u)([-u(2)-u(3);u(1)+(a*u(2));b + u(3)*(u(1)-c)]);    %function handle for ODE
tspan = [0,340];                   %time interval
u0 = [0,-6.78,0.02];               %intial conditions
N = 40;                            %no. of time sub-intervals steps
Ng = 90000;                        %no. of coarse steps
Nf = 4500000;                      %no. of fine steps
epsilon = 10^(-6);                 %error tolerance 
G = 'RK1';                         %coarse solver (see RK.m file)
F = 'RK4';                         %fine solver   (see RK.m file)

% Gaussian Process prior inputs
K = cell(3,1);                                            %cell array stores covariance kernel structure for each output dimension
K{1} = gpcf_sexp('lengthScale',1, 'magnSigma2',1);        %square exponential kernel (isotropic)
K{2} = gpcf_sexp('lengthScale',1, 'magnSigma2',1);        %square exponential kernel (isotropic)
K{3} = gpcf_sexp('lengthScale',1, 'magnSigma2',1);        %square exponential kernel (isotropic)
like = lik_gaussian();                                    %Gaussian likelihood structure

%solve FHN model using GParareal and parareal
[t,u,err,k,gp,~,~] = GParareal(f,tspan,u0,N,Ng,Nf,epsilon,F,G,K,like);
[~,u_p,err_para,k_para] = parareal(f,tspan,u0,N,Ng,Nf,epsilon,F,G);

%run fine solver
n = length(u0);                         % no. of ODE output dimensions
dt = (tspan(2)-tspan(1))/Nf;            % size of fine time step
t_fine = (tspan(1):dt:tspan(end));      % fine time step mesh
[~,F_sol] = RK(t_fine,u0,f,F);          % fine solution


%PLOTS
% Fig 1: GParareal/parareal solutions vs. fine

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
u_prob = vertcat(fine_trajecs1{:,1});            % probabilistic sol
u_para = vertcat(fine_trajecs2{:,1});            % parareal sol

figure(1)
hold on
plot3(F_sol(:,1),F_sol(:,2),F_sol(:,3),'k')
plot3(u_prob((1:(Nf/N):end),1),u_prob((1:(Nf/N):end),2),u_prob((1:(Nf/N):end),3),'ob')
plot3(u_para((1:(Nf/N):end),1),u_para((1:(Nf/N):end),2),u_para((1:(Nf/N):end),3),'*r')
xlabel('$u_1$','Interpreter','latex'); ylabel('$u_2$','Interpreter','latex'); zlabel('$u_3$','Interpreter','latex');
grid on; box on;
view(-75,10)
legend({'Fine','GParareal','Parareal'},'Interpreter','latex','location','northwest')
hold off

% Fig 2: Max absolute errors over succesive iterations 
err(cumsum(cumsum(err~=0)) == 1) = 0; err(end,end) = eps; errs = max(err,[],1); 
err_para(cumsum(cumsum(err_para~=0)) == 1) = 0; err_para(end,end) = eps; errs_para2 = max(err_para,[],1); 

figure(2)
hold on
plot((1:k_para),errs_para2(1:k_para),'-or','LineWidth',1.2);
plot((1:k),errs(1:k),'-ob','LineWidth',1.2);
yline(epsilon,'--k','LineWidth',1.2)
xlabel('$k$','Interpreter','latex'); ylabel('Max. absolute error','Interpreter','latex');
set(gca,'yscale','log')
grid on; box on;
xlim([1,max([k,k_para])]); ylim([10^(-7),10^(4)]);
xticks((1:max([k,k_para]))); %yticks((0:0.5:5));
legend('northeast',{'Parareal','GParareal','Tolerance'})
hold off


% SOLUTION ERRORS: FINE vs. PROBABILISTIC vs. PARAREAL
prob_error = abs(F_sol - u_prob); prob_error(prob_error==0) = 10^(-40);
para_error = abs(F_sol - u_para); para_error(para_error==0) = 10^(-40);


% Fig 3: Absolute errors of the solutions over time 
figure(3)

h(1) = subplot(3,1,1);
hold on
plot(t_fine,para_error(:,1),'r','LineWidth',1.2)
plot(t_fine,prob_error(:,1),'b','LineWidth',1.2)
hold off
set(gca,'xticklabels',[])
yticks(10.^[-15,-10,-5,0])
set(gca,'yscale','log')
ylim([10^(-15) 10^(0)]);
ylabel('$u_1$ error','interpreter','latex');
box on; grid on;

h(2) = subplot(3,1,2);
hold on
plot(t_fine,para_error(:,2),'r','LineWidth',1.2)
plot(t_fine,prob_error(:,2),'b','LineWidth',1.2)
hold off
set(gca,'xticklabels',[])
yticks(10.^[-15,-10,-5,0])
set(gca,'yscale','log')
ylim([10^(-15) 10^(0)]);
ylabel('$u_2$ error','interpreter','latex');
box on; grid on;


h(3) = subplot(3,1,3);
hold on
plot(t_fine,para_error(:,3),'r','LineWidth',1.2)
plot(t_fine,prob_error(:,3),'b','LineWidth',1.2)
hold off
%set(gca,'xtick',[])
yticks(10.^[-15,-10,-5,0])
set(gca,'yscale','log')
ylim([10^(-15) 10^(0)]);
xlabel('$t$','interpreter','latex'); ylabel('$u_3$ error','interpreter','latex');
box on; grid on;
legend('northeast',{'Fine - Parareal','Fine - GParareal'})
