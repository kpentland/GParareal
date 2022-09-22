%% Non-autonomous system (2D system to 3D system)
%Here we solve a pair of non-autonomous coupled ODEs. To run the 
%code, follow the description and instructions in each section.

%WARNING: some of the code here is designed to run on a HPC system with at least 
%512 compute cores and high RAM storage. There will be a note in that section for
%what to do in this case - note that that results may differ between machines
%and parallel speedup may not be realised. 


%% Plots: Figure 4.6(a),(b)
% Solve system and plot results. Additional results shown for diagnostic purposes.  

clear; close all; clc;

% INPUTS
f = @(t,u)([-u(2) + u(1)*((u(3)/500) - u(1)^2 - u(2)^2);u(1) + u(2)*((u(3)/500) - u(1)^2- u(2)^2);1]);  %the ODE to be solved for parareal
tspan = [-20,500];                   %time interval of integration
u0 = [0.1,0.1,tspan(1)];             %initial condition for ODE system + initial time of integration
epsilon = 10^(-6);                   %error tolerance
N = 32;                              %no. of time sub-intervals (processors)
Ng = 2*1024;                         %coarse time steps
Nf = Ng*85;                          %fine time steps
G = 'RK1';                           %coarse solver
F = 'RK8';                           %fine solver


%fine solve
n = length(u0);                         % no. of ODE output dimensions
dt = (tspan(2)-tspan(1))/Nf;            % size of fine time step
t_fine = (tspan(1):dt:tspan(end));      % fine time step mesh
[~,F_sol] = RK(t_fine,u0,f,F);          % fine solution

% Gaussian Process emulator inputs
kernel = struct('func',@isoSE_cov,'theta',[1,1],'jitter',1e-12);

%solve FHN model using GParareal and parareal
[t,u,err,k,kernel,xx,DD,T] = GParareal(f,tspan,u0,N,Ng,Nf,epsilon,F,G,kernel);
[~,u_p,err_para,k_para,slices] = parareal(f,tspan,u0,N,Ng,Nf,epsilon,F,G);


%PLOTS
%Fig 1: PPODE/parareal solutions vs. fine

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
subplot(2,1,1)
hold on
plot(t_fine,F_sol(:,1),'k')
plot(t,u(:,end-2),'ob')
plot(t,u_p(:,end-2),'*r')
hold off
xlim([-20,150]); ylim([-1.5,1.5])
set(gca,'xticklabels',[])
ylabel('$u_1(t)$','interpreter','latex');
set(gca,'fontsize',12)
box on; grid on;

subplot(2,1,2)
hold on
plot(t_fine,F_sol(:,2),'k')
plot(t,u(:,end-1),'ob')
plot(t,u_p(:,end-1),'*r')
hold off
xlim([-20,150]); ylim([-1.5 1.5])
xlabel('$t$','interpreter','latex'); ylabel('$u_2(t)$','interpreter','latex');
set(gca,'fontsize',12)
box on; grid on;

legend({'Fine','GParareal','Parareal'},'Interpreter','latex','numcolumns',3,'location','southwest')
hold off


%Fig 2: Convergence plot (PPODE vs. parareal)

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
xlim([1,28]); ylim([10^(-8),10^(1)]);
xticks((1:2:28)); %yticks((0:0.5:5));
legend({'Parareal','GParareal','Tolerance'},'location','southwest')
hold off

%Fig 3: Error plot (PPODE/parareal vs. fine solution)

%calculate absolute error at each time step (for each algorithm)
prob_error1 = abs(F_sol(:,1) - u_prob(:,1)); prob_error1(prob_error1==0) = 10^(-40);
para_error1 = abs(F_sol(:,1) - u_para(:,1)); para_error1(para_error1==0) = 10^(-40);

prob_error2 = abs(F_sol(:,2) - u_prob(:,2)); prob_error2(prob_error2==0) = 10^(-40);
para_error2 = abs(F_sol(:,2) - u_para(:,2)); para_error2(para_error2==0) = 10^(-40);

figure(3)
subplot(2,1,1)
set(gca,'fontsize',12)
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
set(gca,'fontsize',12)
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


% Fig 4: 3D plot of solution 
figure(4)
hold on
plot3(t_fine,F_sol(:,1),F_sol(:,2),'k')
plot3(u_prob((1:(Nf/N):end),3),u_prob((1:(Nf/N):end),1),u_prob((1:(Nf/N):end),2),'ob')
plot3(u_para((1:(Nf/N):end),3),u_para((1:(Nf/N):end),1),u_para((1:(Nf/N):end),2),'*r')
xlabel('$t$','Interpreter','latex'); ylabel('$u_1$','Interpreter','latex'); zlabel('$u_2$','Interpreter','latex');
grid on; box on;
view(-20,20)
xlim([-20,150])
legend({'Fine','GParareal','Parareal'},'Interpreter','latex','location','northwest')
hold off




%% Plots: Figure 4.7(a),(b) and Table 4.1
% Here we display the results given in Table 4.1 and plotted in Figure 4.7.
% These results are pulled from saved .mat files generated by running
% GParareal and parareal on 512 cores (these are stored in 'Test 2' folder).

% To run this section, you need to add the 'Seclected Folders and
% Subfolders' of 'Test 2' folder to your path. Do this by right-clicking 'Test 2'.

% If interested in running the code for these results, please contact the
% lead author. 

clear; close all; clc;

res = cell(5,17);
res(1,:) = {'J','kgpara','kpara','Tg','Tf','Tgp','Tser','Tpara','Tgpara','Spara','Sgpara','Tpara','Tgpara','Spara','Sgpara','Epara','Egpara'};
% res(2,6:8) = {'Actual'};
res(2,12:15) = {'Theory'};
res(3:7,1) = {'32','64','128','256','512'};


load('GP_NA_N32.mat')
i = 3;
res{i,2} = k;

res{i,4} = T{1};
res{i,5} = T{2};
res{i,6} = sum(T{3},'all'); %max(sum(T{3},1));
res{i,7} = N*T{2};
res{i,8} = 89165.98;          % Tpara (manual)
res{i,9} = 43321.64;          % Tgpara (manual)
res{i,10} = res{i,7}/res{i,8};
res{i,11} = res{i,7}/res{i,9};

res{i,13} = k*T{2} + (k+1)*(N - (k/2))*T{1} + res{i,6};
res{i,15} = 1/((k/N) + (k+1)*(1 - (k/(2*N)))*(T{1}/T{2}) + (1/(N*T{2}))*res{i,6});
res{i,17} = res{i,11}/res{i,15};

load('P_NA_N32.mat')
res{i,3} = k_para;
res{i,12} = k_para*T{2} + (k_para+1)*(N - (k_para/2))*T{1};
res{i,14} = 1/((k_para/N) + (k_para+1)*(1 - (k_para/(2*N)))*(T{1}/T{2}));
res{i,16} = res{i,10}/res{i,14};


load('GP_NA_N64.mat')
i = 4;
res{i,2} = k;

res{i,4} = T{1};
res{i,5} = T{2};
res{i,6} = sum(T{3},'all'); %max(sum(T{3},1));
res{i,7} = N*T{2};
res{i,8} = 67471.71;          % Tpara (manual)
res{i,9} = 31978.64;          % Tgpara (manual)
res{i,10} = res{i,7}/res{i,8};
res{i,11} = res{i,7}/res{i,9};

res{i,13} = k*T{2} + (k+1)*(N - (k/2))*T{1} + res{i,6};
res{i,15} = 1/((k/N) + (k+1)*(1 - (k/(2*N)))*(T{1}/T{2}) + (1/(N*T{2}))*res{i,6});
res{i,17} = res{i,11}/res{i,15};

load('P_NA_N64.mat')
res{i,3} = k_para;
res{i,12} = k_para*T{2} + (k_para+1)*(N - (k_para/2))*T{1};
res{i,14} = 1/((k_para/N) + (k_para+1)*(1 - (k_para/(2*N)))*(T{1}/T{2}));
res{i,16} = res{i,10}/res{i,14};




load('GP_NA_N128.mat')
i = 5;
res{i,2} = k;

res{i,4} = T{1};
res{i,5} = T{2};
res{i,6} = sum(T{3},'all'); %max(sum(T{3},1));
res{i,7} = N*T{2};
res{i,8} = 64744.97;          % Tpara (manual)
res{i,9} = 18962.55;          % Tgpara (manual)
res{i,10} = res{i,7}/res{i,8};
res{i,11} = res{i,7}/res{i,9};

res{i,13} = k*T{2} + (k+1)*(N - (k/2))*T{1} + res{i,6};
res{i,15} = 1/((k/N) + (k+1)*(1 - (k/(2*N)))*(T{1}/T{2}) + (1/(N*T{2}))*res{i,6});
res{i,17} = res{i,11}/res{i,15};

load('P_NA_N128.mat')
res{i,3} = k_para;
res{i,12} = k_para*T{2} + (k_para+1)*(N - (k_para/2))*T{1};
res{i,14} = 1/((k_para/N) + (k_para+1)*(1 - (k_para/(2*N)))*(T{1}/T{2}));
res{i,16} = res{i,10}/res{i,14};


load('GP_NA_N256.mat')
i = 6;
res{i,2} = k;

res{i,4} = T{1};
res{i,5} = T{2};
res{i,6} = sum(T{3},'all'); %max(sum(T{3},1));
res{i,7} = N*T{2};
res{i,8} = 56402.50;          % Tpara (manual)
res{i,9} = 11709.83;          % Tgpara (manual)
res{i,10} = res{i,7}/res{i,8};
res{i,11} = res{i,7}/res{i,9};

res{i,13} = k*T{2} + (k+1)*(N - (k/2))*T{1} + res{i,6};
res{i,15} = 1/((k/N) + (k+1)*(1 - (k/(2*N)))*(T{1}/T{2}) + (1/(N*T{2}))*res{i,6});
res{i,17} = res{i,11}/res{i,15};

load('P_NA_N256.mat')
res{i,3} = k_para;
res{i,12} = k_para*T{2} + (k_para+1)*(N - (k_para/2))*T{1};
res{i,14} = 1/((k_para/N) + (k_para+1)*(1 - (k_para/(2*N)))*(T{1}/T{2}));
res{i,16} = res{i,10}/res{i,14};




load('GP_NA_N512.mat')
i = 7;
res{i,2} = k;

res{i,4} = T{1};
res{i,5} = T{2};
res{i,6} = sum(T{3},'all'); %max(sum(T{3},1));
res{i,7} = N*T{2};
res{i,8} = 44210.60;          % Tpara (manual)
res{i,9} = 20978.78;          % Tgpara (manual)
res{i,10} = res{i,7}/res{i,8};
res{i,11} = res{i,7}/res{i,9};

res{i,13} = k*T{2} + (k+1)*(N - (k/2))*T{1} + res{i,6};
res{i,15} = 1/((k/N) + (k+1)*(1 - (k/(2*N)))*(T{1}/T{2}) + (1/(N*T{2}))*res{i,6});
res{i,17} = res{i,11}/res{i,15};

load('P_NA_N512.mat')
res{i,3} = k_para;
res{i,12} = k_para*T{2} + (k_para+1)*(N - (k_para/2))*T{1};
res{i,14} = 1/((k_para/N) + (k_para+1)*(1 - (k_para/(2*N)))*(T{1}/T{2}));
res{i,16} = res{i,10}/res{i,14};

% prints table
res


figure(1)
set(gca,'fontsize',12)
hold on
plot(log2([32,64,128,256,512]),log2([res{3,12},res{4,12},res{5,12},res{6,12},res{7,12}]),'-*r','LineWidth',1.2);
plot(log2([32,64,128,256,512]),log2([res{3,8},res{4,8},res{5,8},res{6,8},res{7,8}]),'--or','LineWidth',1.2);
plot(log2([32,64,128,256,512]),log2([res{3,13},res{4,13},res{5,13},res{6,13},res{7,13}]),'-*b','LineWidth',1.2);
plot(log2([32,64,128,256,512]),log2([res{3,9},res{4,9},res{5,9},res{6,9},res{7,9}]),'--ob','LineWidth',1.2);
plot(log2([32,64,128,256,512]),log2([res{3,7},res{4,7},res{5,7},res{6,7},res{7,7}]),'--ok','LineWidth',1.2);
xlabel('$J$','Interpreter','latex'); ylabel('$\log_2$(wallclock time)','Interpreter','latex');
xticks((5:10)); xticklabels(2.^(5:10)); xlim([5 9])
yticks((12:18)); yticklabels((12:18)); ylim([12 17.5])
grid on; box on;
legend({'Parareal (theory)','Parareal (numerical)','GParareal (theory)','GParareal (numerical)','Fine Solver (numerical)'},'interpreter','latex','location','southwest')
hold off

figure(2)
set(gca,'fontsize',12)
hold on
plot(log2([32,64,128,256,512]),[res{3,14},res{4,14},res{5,14},res{6,14},res{7,14}],'-*r','LineWidth',1.2);
plot(log2([32,64,128,256,512]),[res{3,10},res{4,10},res{5,10},res{6,10},res{7,10}],'--or','LineWidth',1.2);
plot(log2([32,64,128,256,512]),[res{3,15},res{4,15},res{5,15},res{6,15},res{7,15}],'-*b','LineWidth',1.2);
plot(log2([32,64,128,256,512]),[res{3,11},res{4,11},res{5,11},res{6,11},res{7,11}],'--ob','LineWidth',1.2);
plot(log2([32 1024]),[1 1],'--k','LineWidth',1.2)
xlabel('$J$','interpreter','latex'); ylabel('Speedup','interpreter','latex');
xticks((5:10)); xticklabels(2.^(5:10)); xlim([5 9])
ylim([0 15]); yticks((0:2:14))
grid on; box on;
legend({'Parareal (theory)','Parareal (numerical)','GParareal (theory)','GParareal (numerical)','Fine Solver'},'interpreter','latex','location','northwest')
hold off

