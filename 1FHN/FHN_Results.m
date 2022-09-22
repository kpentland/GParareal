%% FitzHugh-Nagumo model
%Here we solve the FHN system - a pair of coupled stiff ODEs. To run the 
%code, follow the description and instructions in each section.

%WARNING: some of the code here is designed to run on a HPC system with at least 
%40 compute cores and high RAM storage. There will be a note in that section for
%what to do in this case - note that that results may differ between machines
%and parallel speedup may not be realised. 

%% Solve system
%This section solves the system using GParareal and parareal, saving the
%solutions for use later. Saving turned off to avoid overwriting saved
%results. 

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
Nf = 160000;                    %fine time steps
G = 'RK2';                         %coarse solver (see RK.m file)
F = 'RK4';                         %fine solver   (see RK.m file)

% Gaussian Process emulator inputs
kernel = struct('func',@isoSE_cov,'theta',[1,1],'jitter',10^(-14));

%solve FHN model using GParareal and parareal
[t,u,err,k,gp_kers,xx,DD] = GParareal(f,tspan,u0,N,Ng,Nf,epsilon,F,G,kernel);
[~,u_p,err_para,k_para] = parareal(f,tspan,u0,N,Ng,Nf,epsilon,F,G);

% display optimised GP hyperparameters from GParareal (note they are in log
% form)
fprintf('Optimised GP hyperparameters - output one: \n')
t1 = gp_kers{1}.theta;  disp(t1)
fprintf('Optimised GP hyperparameters - output two: \n')
t2 = gp_kers{2}.theta;  disp(t2)

%run fine solver
n = length(u0);                         % no. of ODE output dimensions
dt = (tspan(2)-tspan(1))/Nf;            % size of fine time step
t_fine = (tspan(1):dt:tspan(end));      % fine time step mesh
[~,F_sol] = RK(t_fine,u0,f,F);          % fine solution

% save solutions
save('FHN_solutions.mat')


%% Plots: Figures 4.1(a),(b),(c)
% In this section we load the solutions from above and plot. 

clear; close all; clc;

load('FHN_solutions.mat')


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
set(gca,'fontsize',12)
box on; grid on;

subplot(2,1,2)
hold on
plot(t_fine,F_sol(:,2),'k')
plot(t_fine((1:(Nf/N):end)),u_GPara((1:(Nf/N):end),2),'ob')
plot(t_fine((1:(Nf/N):end)),u_para((1:(Nf/N):end),2),'*r')
hold off
xlim(tspan); ylim([-1.5 1.5])
xlabel('$t$','interpreter','latex'); ylabel('$u_2(t)$','interpreter','latex');
set(gca,'fontsize',12)
box on; grid on;

legend({'Fine','GParareal','Parareal'},'Interpreter','latex','numcolumns',3,'location','southwest')
hold off

%print(gcf,'FHN_sols.png','-dpng','-r500')



%Fig 2: Convergence plot (GParareal vs. parareal)

%calculate maximum error during each iteration (for each algorithm)
err(cumsum(cumsum(err~=0)) == 1) = 0; err(end,end) = eps; errs = max(err,[],1); 
err_para(cumsum(cumsum(err_para~=0)) == 1) = 0; err_para(end,end) = eps; errs_para = max(err_para,[],1); 

figure(2)
set(gca,'fontsize',12)
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


%% Parallel speedup 

% WARNING: Here we calculate runtimes for the problem using up to 40 cores. 
% This section can take a long time to run so do so at your own risk! Skip
% to the next section to plot the pre-run results. Results saving is
% comment out below. 

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
Nf = 160000000;                    %fine time steps 
G = 'RK2';                         %coarse solver (see RK.m file)
F = 'RK4';                         %fine solver   (see RK.m file)

% Gaussian Process emulator inputs
kernel = struct('func',@isoSE_cov,'theta',[1,1],'jitter',10*eps);

% number of different cores to run on 
cores = [1,2,4,8,16,32,40]';
%no. of simulations to average wallclock time over
runs = 5;

%storage timings in these matrices
GParareal_times = zeros(length(cores),runs+1); GParareal_times(:,1) = cores;
para_times = zeros(length(cores),runs+1);  para_times(:,1) = cores;

for i = length(cores):-1:1
    parpool(cores(i))   %turns on parallel pool
    
    for j = 2:runs+1
        T = tic;
        [~,~,~,~,~,~,~] = GParareal(f,tspan,u0,N,Ng,Nf,epsilon,F,G,kernel);
        s = toc(T);
        GParareal_times(i,j) = s;
        
        T = tic;
        [~,~,~,~] = parareal(f,tspan,u0,N,Ng,Nf,epsilon,F,G);
        s = toc(T);
        para_times(i,j) = s;
    end
    
    delete(gcp)
    save('FHN_timings.mat')
end

% fine time mesh
dt = (tspan(2)-tspan(1))/Nf;
t_fine = (tspan(1):dt:tspan(end));

% time the full serial fine solve
F_times = zeros(runs,1);
for j = 1:runs
    T = tic;
    [~,~] = RK(t_fine,u0,f,F);
    s = toc(T);
    F_times(j,1) = s;
end
clear t_fine

% save('FHN_timings.mat')

%% Plots: Figure 4.2(d)
% Here we plot the results from the benchmarking above. 

clear; close all; clc;

load('FHN_timings.mat')

%calculate average times for each number of processors
avg_GParareal = median(GParareal_times(:,2:end),2);
avg_para = median(para_times(:,2:end),2);
fine_time = median(F_times);

%Fig. 1: average run times against processors used (+ corresponding
% speed-up in inset plot)
figure(3)
set(gca,'fontsize',12)
hold on
plot(log2(GParareal_times(:,1)),log2(avg_para),'-or','LineWidth',1.2)
plot(log2(GParareal_times(:,1)),log2(avg_GParareal),'-ob','LineWidth',1.2)
plot(log2([1 64]),log2([fine_time fine_time]),'--k','LineWidth',1.2)
xlabel('Processors','interpreter','latex'); ylabel('$\log_2$(wallclock time)','interpreter','latex');
xticklabels(2.^(0:6))
ylim([6.5 14.5])
grid on; box on;
legend({'Parareal','GParareal','Fine Solver'},'interpreter','latex','location','southwest')
hold off

axes('Position',[.445 .58 .43 .31]);
set(gca,'fontsize',12)
hold on
plot(log2(GParareal_times(:,1)),fine_time./avg_para,'-or','LineWidth',1.2)
plot(log2(GParareal_times(:,1)),fine_time./avg_GParareal,'-ob','LineWidth',1.2)
plot(log2([1 64]),[1 1],'--k','LineWidth',1.2)
xlabel('Processors','interpreter','latex'); ylabel('Speedup','interpreter','latex');
xticks((0:6))
yticks((0:6))
xticklabels(2.^(0:6))
grid on; box on;
hold off

%% Solve system for various ICs (heat maps)
% This section runs GParareal and parareal over many initial values to compare
% convergence rates in two heat maps. WARNING: takes a long time to run!
% Skip to next section to plot results. Solution saving turned off. 

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
Nf = 160000000;                    %fine time steps
G = 'RK2';                         %coarse solver (see RK.m file)
F = 'RK4';                         %fine solver   (see RK.m file)

kernel = struct('func',@isoSE_cov,'theta',[1,1],'jitter',10*eps);

% set of ICs to run
u1 = linspace(-1.25,1.25,11);
u2 = linspace(1.25,-1.25,11);

%store convergence rates for each simulation
k_GParareal = zeros(length(u2),length(u1));
k_para = k_GParareal;

% find each convergence rate (GParareal)
for i = 1:length(u2)
    for j = 1:length(u1)
        [~,~,~,k,~,~,~] = GParareal(f,tspan,[u1(j),u2(i)],N,Ng,Nf,epsilon,F,G,kernel);
        k_GParareal(i,j) = k;
    end
end

% save('FHN_heatmap.mat')


%% Plots: Figures 4.2(a),(b) 
% Here we plot the heatmaps described in the prior section. 

clear; close all; clc

load('FHN_heatmap.mat')

%fix colormap
B = jet;

%plot the heatmap (parareal)
figure(1)
h2 = heatmap(u1,u2,k_para);
h2.ColorLimits = [10,15];
colormap(B(156:end,:))
colorbar off
h2.XLabel = '$u_1(t=0)$';
h2.YLabel = '$u_2(t=0)$';
h2.NodeChildren(3).XAxis.Label.Interpreter = 'latex';
h2.NodeChildren(3).YAxis.Label.Interpreter = 'latex';


%plot the heatmap (GParareal)
figure(2)
h1 = heatmap(u1,u2,k_GParareal);
h1.ColorLimits = [2,6];
colormap(B(1:80,:))
colorbar off
h1.XLabel = '$u_1(t=0)$';
h1.YLabel = '$u_2(t=0)$';
h1.NodeChildren(3).XAxis.Label.Interpreter = 'latex';
h1.NodeChildren(3).YAxis.Label.Interpreter = 'latex';



%% Solve system using legacy data
%Here we solve the FHN system for an IC given legacy data from a prior
%simulation (that used a different IC). Can be skipped, results plotted in 
%next section. Results saving turned off. 

clear; close all; clc;

%load legacy data from section one
load('FHN_solutions.mat')

%define new IC (to solve for with and without legacy data)
u0_new = [0.75,0.25];

%solve problem without legacy data
[~,u1,err1,k,~,~,~] = GParareal(f,tspan,u0_new,N,Ng,Nf,epsilon,F,G,kernel);
%solve problem with legacy data
[~,u2,err2,k2,~,~,~] = GParareal(f,tspan,u0_new,N,Ng,Nf,epsilon,F,G,kernel,xx,DD);
%solve using fine solver and parareal (for comparison)
[~,F_sol2] = RK(t_fine,u0_new,f,F);                                                            
[~,u_p,err_para,k_para] = parareal(f,tspan,u0_new,N,Ng,Nf,epsilon,F,G);                

%save('FHN_solutions_using_legacy.mat')

%% Plots: Figures 4.3(a),(b)
%Here plot the convergence rates of each simulation from the prior section
%and their numerical errors with respect to the fine solution. 

clear; close all; clc;

%load solutions
load('FHN_solutions_using_legacy.mat')


% use the ICs found by each algorithm to integrate in parallel
fine_trajecs3 = cell(N,1);
fine_trajecs4 = cell(N,1);
fine_trajecs5 = cell(N,1);
for i = 1:N
    [~,temp3] = RK((t(i):dt:t(i+1)),u1(i,(n*(k-1)+1:n*k)),f,F);
    [~,temp4] = RK((t(i):dt:t(i+1)),u2(i,(n*(k2-1)+1:n*k2)),f,F);
    [~,temp5] = RK((t(i):dt:t(i+1)),u_p(i,(n*(k_para-1)+1:n*k_para)),f,F);
    if i < N
        fine_trajecs3{i,1} = temp3(1:end-1,:);
        fine_trajecs4{i,1} = temp4(1:end-1,:);
        fine_trajecs5{i,1} = temp5(1:end-1,:);
    else
        fine_trajecs3{i,1} = temp3;
        fine_trajecs4{i,1} = temp4;
        fine_trajecs5{i,1} = temp5;
    end
end
u_GPara2 = vertcat(fine_trajecs3{:,1});            % GParareal solution (without legacy)
u_GPara3 = vertcat(fine_trajecs4{:,1});            % GParareal solution (with legacy)
u_para2 = vertcat(fine_trajecs5{:,1});             % parareal soluiton


% PLOTS

% Fig 1: Max absolute errors over succesive iterations 
err1(cumsum(cumsum(err1~=0)) == 1) = 0; err1(end,end) = eps; errs2 = max(err1,[],1); 
err2(cumsum(cumsum(err2~=0)) == 1) = 0; err2(end,end) = eps; errs3 = max(err2,[],1); 
err_para(cumsum(cumsum(err_para~=0)) == 1) = 0; err_para(end,end) = eps; errs_para2 = max(err_para,[],1); 

figure(1)
hold on
plot((1:k_para),errs_para2(1:k_para),'-or','LineWidth',1.2);
plot((1:k),errs2(1:k),'-ob','LineWidth',1.2);
plot((1:k2),errs3(1:k2),'color','#77AC30','marker','o','LineWidth',1.2);
yline(epsilon,'--k','LineWidth',1.2)
xlabel('$k$','Interpreter','latex'); ylabel('Max. absolute error','Interpreter','latex');
set(gca,'yscale','log')
grid on; box on;
xlim([1,max([k,k2,k_para])]); ylim([10^(-8),10^(2)]);
xticks((1:max([k,k2,k_para]))); %yticks((0:0.5:5));
legend('northeast',{'Parareal','GParareal (no legacy)','GParareal (legacy)','Tolerance'})
hold off


% SOLUTION ERRORS: FINE vs. GParareal vs. PARAREAL
prob_error2 = abs(F_sol2 - u_GPara2); prob_error2(prob_error2==0) = 10^(-40);
prob_error3 = abs(F_sol2 - u_GPara3); prob_error3(prob_error3==0) = 10^(-40);
para_error2 = abs(F_sol2 - u_para2); para_error2(para_error2==0) = 10^(-40);


% Fig 2: Absolute errors of the solutions over time 
figure(2)

subplot(2,1,1)
hold on
plot(t_fine,para_error2(:,1),'r','LineWidth',1.2)
plot(t_fine,prob_error2(:,1),'b','LineWidth',1.2)
plot(t_fine,prob_error3(:,1),'color','#77AC30','LineWidth',1.2)
hold off
set(gca,'xticklabels',[])
yticks(10.^[-15,-10,-5,0])
set(gca,'yscale','log')
ylim([10^(-15) 10^(0)]);
ylabel('$u_1$ error','interpreter','latex');
box on; grid on;

subplot(2,1,2)
hold on
plot(t_fine,para_error2(:,2),'r','LineWidth',1.2)
plot(t_fine,prob_error2(:,2),'b','LineWidth',1.2)
plot(t_fine,prob_error3(:,2),'color','#77AC30','LineWidth',1.2)
hold off
%set(gca,'xtick',[])
yticks(10.^[-15,-10,-5,0])
set(gca,'yscale','log')
ylim([10^(-15) 10^(0)]);
xlabel('$t$','interpreter','latex'); ylabel('$u_2$ error','interpreter','latex');
box on; grid on;
legend('northeast',{'Fine - Parareal','Fine - GParareal (no legacy)','Fine - GParareal (legacy)','Tolerance'})


%% Solve system for various ICs using legacy data (heat map)
% This section runs GParareal over many initial values to calculate
% convergence rates in a heat map (using the legacy data from the first 
% section).

clear; close all; clc;

%load legacy data from section one
load('FHN_solutions.mat')

% set of ICs to run
u1 = linspace(-1.25,1.25,11);
u2 = linspace(1.25,-1.25,11);

%store convergence rates for each IVP
k_GParareal_legacy = zeros(length(u2),length(u1));

% solve for the convergence rates using legacy data (GParareal only)
for i = 1:length(u2)
    for j = 1:length(u1)
        [~,~,~,k,~,~,~] = GParareal(f,tspan,[u1(j),u2(i)],N,Ng,Nf,epsilon,F,G,kernel,xx,DD);
        k_GParareal_legacy(i,j) = k;
    end
end

% save('FHN_legacy_heatmap.mat')

%% Plot: Figure 4.4

clear; close all; clc

load('FHN_legacy_heatmap.mat')

% fix colormap
B = jet;

%plot the heatmap 
figure(1)
h1 = heatmap(u1,u2,k_GParareal_legacy-k_GParareal);
% h1.ColorLimits = [2,6];
colormap(B(1:80,:))
colorbar off
h1.XLabel = '$u_1(t=0)$';
h1.YLabel = '$u_2(t=0)$';
h1.NodeChildren(3).XAxis.Label.Interpreter = 'latex';
h1.NodeChildren(3).YAxis.Label.Interpreter = 'latex';
