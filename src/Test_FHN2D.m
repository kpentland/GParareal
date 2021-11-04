%% FitzHugh-Nagumo system (2D)
%Here we solve the FHN system - another pair of coupled stiff ODEs. Plots are described below.  
%To test, run this section. 

clear; close all; clc;

% INPUTS
%initialise the FHN model
a = 0.2;
b = 0.2;
c = 3;
u0 = [-1,1];                     %initial condition for ODE
f = @(t,u)([c*(u(1) - ((u(1)^3)/3) + u(2)); -(1/c)*(u(1) - a + b*u(2))]);    %the ODE to be solved
auto = 1;                          %is the system autonomous (yes=1, no=0)
tspan = [0,40];                    %time interval of integration
epsilon = 10^(-6);                 %error tolerance
N = 40;                            %no. of time sub-intervals (processors)
Ng = N*4;                          %coarse time steps
Nf = Ng*250000;                      %fine time steps
G = 'RK2';                         %coarse solver
F = 'RK4';                         %fine solver

% Gaussian Process prior inputs
K = cell(2,1);                                          %covariance kernel structure required for each output
K{1} = gpcf_sexp('lengthScale',1, 'magnSigma2',1);      %square exponential kernel (isotropic)
K{2} = gpcf_sexp('lengthScale',1, 'magnSigma2',1);      %square exponential kernel (isotropic)
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

subplot(2,1,1)
hold on
plot(t_fine,F_sol(:,1),'k')
plot(t_fine((1:(Nf/N):end)),u_prob((1:(Nf/N):end),1),'ob')
plot(t_fine((1:(Nf/N):end)),u_para((1:(Nf/N):end),1),'*r')
hold off
xlim(tspan); ylim([-3 3])
xlabel('$t$','interpreter','latex'); ylabel('$u_1(t)$','interpreter','latex');
box on; grid on;

subplot(2,1,2)
hold on
plot(t_fine,F_sol(:,2),'k')
plot(t_fine((1:(Nf/N):end)),u_prob((1:(Nf/N):end),2),'ob')
plot(t_fine((1:(Nf/N):end)),u_para((1:(Nf/N):end),2),'*r')
hold off
xlim(tspan); ylim([-1.5 1.5])
xlabel('$t$','interpreter','latex'); ylabel('$u_2(t)$','interpreter','latex');
box on; grid on;

legend({'Fine','PPODE','Parareal'},'Interpreter','latex','numcolumns',3,'location','southwest')
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
xlabel('$k$','Interpreter','latex'); ylabel('Max. absolute error','Interpreter','latex');
set(gca,'yscale','log')
grid on; box on;
xlim([1,max([k,k_para])]); ylim([10^(-8),10^(2)]);
xticks((1:max([k,k_para]))); %yticks((0:0.5:5));
legend('northeast',{'Parareal','PPODE','Tolerance'})
hold off

%Fig 3: Error plot (PPODE/parareal vs. fine solution)

%calculate absolute error at each time step (for each algorithm)
prob_error1 = abs(F_sol(:,1) - u_prob(:,1)); prob_error1(prob_error1==0) = 10^(-40);
para_error1 = abs(F_sol(:,1) - u_para(:,1)); para_error1(para_error1==0) = 10^(-40);

prob_error2 = abs(F_sol(:,2) - u_prob(:,2)); prob_error2(prob_error2==0) = 10^(-40);
para_error2 = abs(F_sol(:,2) - u_para(:,2)); para_error2(para_error2==0) = 10^(-40);

figure(3)
subplot(2,1,1)
hold on
plot(t_fine,para_error1,'r','LineWidth',1.2)
plot(t_fine,prob_error1,'b','LineWidth',1.2)
hold off
set(gca,'yscale','log')
xlim(tspan)
ylim([10^(-15) 10^(0)]);
xlabel('$t$','interpreter','latex'); ylabel('$u_1$ error','interpreter','latex');
box on; grid on;
legend({'Fine - Parareal','Fine - PPODE'},'interpreter','latex','location','northwest','numcolumns',2)

subplot(2,1,2)
hold on
plot(t_fine,para_error2,'r','LineWidth',1.2)
plot(t_fine,prob_error2,'b','LineWidth',1.2)
hold off
set(gca,'yscale','log')
xlim(tspan)
ylim([10^(-15) 10^(0)]);
xlabel('$t$','interpreter','latex'); ylabel('$u_2$ error','interpreter','latex');
box on; grid on;

%Fig 4/5: mean of the GP regression over (F-G)(x) (for each output dimension)

% Choose test points on a dense grid
ns = 5625;
xs1 = linspace(-3,3,sqrt(ns));
xs2 = linspace(-1.5,1.5,sqrt(ns));
[xs1,xs2] = meshgrid(xs1,xs2);

%predict the means
indep_means = cell(n,1);
for j = 1:n
    [indep_means{j},~] = gp_pred(gp{j},xx,DD(:,j),[xs1(:),xs2(:)]);
end

% Plot the predictions and data
figure(4)
clf
mesh(xs1, xs2, reshape(indep_means{1},sqrt(ns),sqrt(ns)));
hold on
plot3(xx(:,1),xx(:,2),DD(:,1),'.r')
axis([-3 3 -1.5 1.5 -0.5 0.5])
axis on;
xlabel('$u_1$','interpreter','latex'); ylabel('$u_2$','interpreter','latex');
zlabel('$(\mathcal{F}-\mathcal{G})_1 (u_1,u_2)$','interpreter','latex');
title('Output one');

figure(5)
clf
mesh(xs1, xs2, reshape(indep_means{2},sqrt(ns),sqrt(ns)));
hold on
plot3(xx(:,1),xx(:,2),DD(:,2),'.r')
axis([-3 3 -1.5 1.5 -0.03 0.03])
axis on;
xlabel('$u_1$','interpreter','latex'); ylabel('$u_2$','interpreter','latex');
zlabel('$(\mathcal{F}-\mathcal{G})_2 (u_1,u_2)$','interpreter','latex');
title('Output two');


%% Test area: Heat maps
% This section runs PPODE and parareal over many initial values to compare
% convergence rates. 

clear; close all; clc;

% INPUTS
%initialise the FHN model
a = 0.2;
b = 0.2;
c = 3;
f = @(t,u)([c*(u(1) - ((u(1)^3)/3) + u(2)); -(1/c)*(u(1) - a + b*u(2))]);    %the ODE to be solved
auto = 1;                          %is the system autonomous (yes=1, no=0)
tspan = [0,40];                    %time interval of integration
epsilon = 10^(-6);                 %error tolerance
N = 40;                            %no. of time sub-intervals (processors)
Ng = N*4;                          %coarse time steps
Nf = Ng*250000;                       %fine time steps
G = 'RK2';                         %coarse solver
F = 'RK4';                         %fine solver

% Gaussian Process prior inputs
K = cell(2,1);                                          %covariance kernel structure required for each output
K{1} = gpcf_sexp('lengthScale',1, 'magnSigma2',1);      %square exponential kernel (isotropic)
K{2} = gpcf_sexp('lengthScale',1, 'magnSigma2',1);      %square exponential kernel (isotropic)
like = lik_gaussian();                                  %Gaussian likelihood with prior scale parameter

% set of ICs to run
u1 = linspace(-1.5,1.5,13);
u2 = linspace(1.5,-1.5,13);

%store convergence rates for each IVP
k_PPODE = zeros(length(u2),length(u1));
k_para = k_PPODE;

% solve for the convergence rates (PPODE)
for i = 1:length(u2)
    for j = 1:length(u1)
        [~,~,~,k,~,~,~] = PPODE(f,auto,tspan,[u1(j),u2(i)],N,Ng,Nf,epsilon,F,G,K,like);     % PPODE solution
        k_PPODE(i,j) = k;
    end
end
save('heatmap.mat')

% solve for the convergence rates (parareal)
for i = 1:length(u1)
    for j = 1:length(u2)
        [~,~,~,k] = parareal(f,tspan,[u1(j),u2(i)],N,Ng,Nf,epsilon,F,G);                    % parareal solution
        k_para(i,j) = k;
    end
end
save('heatmap.mat')

%plot the heatmaps
figure(1)
h1 = heatmap(u1,u2,k_PPODE);
h1.ColorLimits = [k_min,k_max];
colormap(jet)
h1.XLabel = '$u_1(t=0)$';
h1.YLabel = '$u_2(t=0)$';
h1.NodeChildren(3).XAxis.Label.Interpreter = 'latex';
h1.NodeChildren(3).YAxis.Label.Interpreter = 'latex';
k_min = min(min(k_PPODE,[],'all'),min(k_para,[],'all'));
k_max = max(max(k_PPODE,[],'all'),max(k_para,[],'all'));

%plot the heatmap
figure(2)
h2 = heatmap(u1,u2,k_para);
h2.ColorLimits = [k_min,k_max];
colormap(jet)
h2.XLabel = '$u_1(t=0)$';
h2.YLabel = '$u_2(t=0)$';
h2.NodeChildren(3).XAxis.Label.Interpreter = 'latex';
h2.NodeChildren(3).YAxis.Label.Interpreter = 'latex';


%% Test area: Speed tests (benchmarking)
%Here we calculate runtimes for the FHN model using up to 40 cores. Note:
%this section requires 40 compute cores and will take hours to run. Do so
%at your own risk!

clear; close all; clc;

% INPUTS
%initialise the FHN model
a = 0.2;
b = 0.2;
c = 3;
u0 = [-1,1];                     %initial condition for ODE
f = @(t,u)([c*(u(1) - ((u(1)^3)/3) + u(2)); -(1/c)*(u(1) - a + b*u(2))]);    %the ODE to be solved
auto = 1;                          %is the system autonomous (yes=1, no=0)
tspan = [0,40];                    %time interval of integration
epsilon = 10^(-6);                 %error tolerance
N = 40;                            %no. of time sub-intervals (processors)
Ng = N*4;                          %coarse time steps
Nf = Ng*250000;                    %fine time steps
G = 'RK2';                         %coarse solver
F = 'RK4';                         %fine solver


% number of different cores to run on 
cores = [1,2,4,8,16,32,40]';
runs = 5;
PPODE_times = zeros(length(cores),runs+1); PPODE_times(:,1) = cores;
para_times = zeros(length(cores),runs+1);  para_times(:,1) = cores;

for i = length(cores):-1:1
    parpool(cores(i))
    
    for j = 2:runs+1
        T = tic;
        [~,~,~,~,~,~,~] = PPODE(f,auto,tspan,u0,N,Ng,Nf,epsilon,F,G,K,like);
        s = toc(T);
        PPODE_times(i,j) = s;
        
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

F_times = zeros(runs,1);
for j = 1:runs
    T = tic;
    [~,~] = RK(t_fine,u0,f,F);
    s = toc(T);
    F_times(j,1) = s;
end
clear t_fine
save('FHN_timings.mat')




avg_PPODE = median(PPODE_times(:,2:end),2);
avg_para = median(para_times(:,2:end),2);
fine_time = median(F_times);

%Fig. 1: average run times against processors used
figure(1)
hold on
plot(log2(PPODE_times(:,1)),log2(avg_PPODE),'-ob','LineWidth',1.3)
plot(log2(PPODE_times(:,1)),log2(avg_para),'-or','LineWidth',1.3)
plot(log2([1 64]),log2([fine_time fine_time]),'--k','LineWidth',1.3)
% plot(log2([1 64]),log2([173.7 173.7]),'--b','LineWidth',1.3)
% plot(log2([1 64]),log2([212.3 212.3]),'--r','LineWidth',1.3)
xlabel('Processors','interpreter','latex'); ylabel('$\log_2$(runtime)','interpreter','latex');
%set(gca, 'YScale', 'log')
%ylim([100,10000])
xticklabels(2.^(0:6))
grid on; box on;
legend({'PPODE','Parareal','Fine Solver'},'interpreter','latex')

%Fig. 2: average speed-up (vs. serial solve) against processors used
figure(2)
hold on
plot(log2(PPODE_times(:,1)),fine_time./avg_PPODE,'-ob','LineWidth',1.3)
plot(log2(PPODE_times(:,1)),fine_time./avg_para,'-or','LineWidth',1.3)
plot(log2([1 64]),[1 1],'--k','LineWidth',1.3)
% plot(log2([1 64]),[4.44 4.44],'--b','LineWidth',1.3)
% plot(log2([1 64]),[3.64 3.64],'--r','LineWidth',1.3)
xlabel('Processors','interpreter','latex'); ylabel('Speedup factor','interpreter','latex');
xticklabels(2.^(0:6))
grid on; box on;
legend({'PPODE','Parareal'},'interpreter','latex','location','northwest')


%Fig. 3: average run times against processors used (+ corresponding
%speed-up)
figure(3)
hold on
plot(log2(PPODE_times(:,1)),log2(avg_PPODE),'-ob','LineWidth',1.3)
plot(log2(PPODE_times(:,1)),log2(avg_para),'-or','LineWidth',1.3)
plot(log2([1 64]),log2([fine_time fine_time]),'--k','LineWidth',1.3)
% plot(log2([1 64]),log2([173.7 173.7]),'--b','LineWidth',1.3)
% plot(log2([1 64]),log2([212.3 212.3]),'--r','LineWidth',1.3)
xlabel('Processors','interpreter','latex'); ylabel('$\log_2$(runtime)','interpreter','latex');
xticklabels(2.^(0:6))
ylim([5.25 11.25])
grid on; box on;
legend({'PPODE','Parareal','Fine Solver'},'interpreter','latex','location','southwest')
hold off

axes('Position',[.445 .58 .43 .31]);
hold on
plot(log2(PPODE_times(:,1)),fine_time./avg_PPODE,'-ob','LineWidth',1.3)
plot(log2(PPODE_times(:,1)),fine_time./avg_para,'-or','LineWidth',1.3)
plot(log2([1 64]),[1 1],'--k','LineWidth',1.3)
xlabel('Processors','interpreter','latex'); ylabel('Speedup','interpreter','latex');
xticks((0:6))
yticks((0:4))
xticklabels(2.^(0:6))
grid on; box on;
hold off


%% Test area: alternate ICs (using prior data)
%Here we solve the FHN system for one IC given prior data from solving for
% an alternate IC. Plots are described below. To test, run this section. 

clear; close all; clc;

% INPUTS
%initialise the FHN model
a = 0.2;
b = 0.2;
c = 3;
u0 = [-1,1];                     %initial condition for ODE
f = @(t,u)([c*(u(1) - ((u(1)^3)/3) + u(2)); -(1/c)*(u(1) - a + b*u(2))]);    %the ODE to be solved
auto = 1;                          %is the system autonomous (yes=1, no=0)
tspan = [0,40];                    %time interval of integration
epsilon = 10^(-6);                 %error tolerance
N = 40;                            %no. of time sub-intervals (processors)
Ng = N*4;                          %coarse time steps
Nf = Ng*500;                       %fine time steps
G = 'RK2';                         %coarse solver
F = 'RK4';                         %fine solver

% Gaussian Process prior inputs
K = cell(2,1);                                          %covariance kernel structure required for each output
K{1} = gpcf_sexp('lengthScale',1, 'magnSigma2',1);      %square exponential kernel (isotropic)
K{2} = gpcf_sexp('lengthScale',1, 'magnSigma2',1);      %square exponential kernel (isotropic)
like = lik_gaussian();                                  %Gaussian likelihood with prior scale parameter

%fine time step and mesh (n = no. of dimensions)
n = length(u0);
dt = (tspan(2)-tspan(1))/Nf;
t_fine = (tspan(1):dt:tspan(end));


%solve problem with original IC
[t,u,err,k,gp,xx,DD] = PPODE(f,auto,tspan,u0,N,Ng,Nf,epsilon,F,G,K,like);     % PPODE solution
[~,F_sol] = RK(t_fine,u0,f,F);                                                % fine solution
[~,u_p,err_para,k_para] = parareal(f,tspan,u0,N,Ng,Nf,epsilon,F,G);           % parareal solution

%solve with alternate IC (using perviously obtained data and without)
u0_new = [-1.5,-1.5];

%solve problem using without prior data
[~,u2,err2,k2,~,~,~] = PPODE(f,auto,tspan,u0_new,N,Ng,Nf,epsilon,F,G,K,like);                  % PPODE solution
%solve problem using previously obtained data
[~,u3,err3,k3,~,~,~] = PPODE(f,auto,tspan,u0_new,N,Ng,Nf,epsilon,F,G,K,like,xx,DD);            % PPODE solution
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
    [~,temp1] = RK((t(i):dt:t(i+1)),u(i,(n*(k-1)+1:n*k)),f,F);
    [~,temp2] = RK((t(i):dt:t(i+1)),u_p(i,(n*(k_para-1)+1:n*k_para)),f,F);
    [~,temp3] = RK((t(i):dt:t(i+1)),u2(i,(n*(k2-1)+1:n*k2)),f,F);
    [~,temp4] = RK((t(i):dt:t(i+1)),u3(i,(n*(k3-1)+1:n*k3)),f,F);
    [~,temp5] = RK((t(i):dt:t(i+1)),u_p2(i,(n*(k_para2-1)+1:n*k_para2)),f,F);
    if i < N
        fine_trajecs1{i,1} = temp1(1:end-1,:);
        fine_trajecs2{i,1} = temp2(1:end-1,:);
        fine_trajecs3{i,1} = temp3(1:end-1,:);
        fine_trajecs4{i,1} = temp4(1:end-1,:);
        fine_trajecs5{i,1} = temp5(1:end-1,:);
    else
        fine_trajecs1{i,1} = temp1;
        fine_trajecs2{i,1} = temp2;
        fine_trajecs3{i,1} = temp3;
        fine_trajecs4{i,1} = temp4;
        fine_trajecs5{i,1} = temp5;
    end
end
u_prob = vertcat(fine_trajecs1{:,1});             % prob solution (old IC)
u_para = vertcat(fine_trajecs2{:,1});             % parareal sol (old IC)
u_prob2 = vertcat(fine_trajecs3{:,1});            % prob solution (new IC - without data)
u_prob3 = vertcat(fine_trajecs4{:,1});            % prob solution (new IC - with data)
u_para2 = vertcat(fine_trajecs5{:,1});            % parareal sol (new IC)




% PLOTS



% SUCCESSIVE ITERATION ERRORS 
err(cumsum(cumsum(err~=0)) == 1) = 0; err(end,end) = eps; errs = max(err,[],1); 
err_para(cumsum(cumsum(err_para~=0)) == 1) = 0; err_para(end,end) = eps; errs_para = max(err_para,[],1); 
err2(cumsum(cumsum(err2~=0)) == 1) = 0; err2(end,end) = eps; errs2 = max(err2,[],1); 
err3(cumsum(cumsum(err3~=0)) == 1) = 0; err3(end,end) = eps; errs3 = max(err3,[],1); 
err_para2(cumsum(cumsum(err_para2~=0)) == 1) = 0; err_para2(end,end) = eps; errs_para2 = max(err_para2,[],1); 


figure(1)
hold on
plot((1:k_para2),errs_para2(1:k_para2),'-ok','LineWidth',1.2);
plot((1:k2),errs2(1:k2),'-ob','LineWidth',1.2);
plot((1:k3),errs3(1:k3),'--ob','LineWidth',1.2);
yline(epsilon,'--k')
xlabel('$k$','Interpreter','latex'); ylabel('Max. absolute error','Interpreter','latex');
set(gca,'yscale','log')
grid on; box on;
xlim([1,max([k2,k3,k_para2])]); ylim([10^(-8),10^(2)]);
xticks((1:max([k2,k3,k_para2]))); %yticks((0:0.5:5));
legend('northeast',{'Parareal','PPODE (no data)','PPODE (with data)','Tolerance'})
hold off


% SOLUTION ERRORS: FINE vs. PROBABILISTIC vs. PARAREAL
prob_error2 = abs(F_sol2 - u_prob2); prob_error2(prob_error2==0) = 10^(-40);
prob_error3 = abs(F_sol2 - u_prob3); prob_error3(prob_error3==0) = 10^(-40);
para_error2 = abs(F_sol2 - u_para2); para_error2(para_error2==0) = 10^(-40);


figure(2)

subplot(2,1,1)
hold on
plot(t_fine,para_error2(:,1),'r','LineWidth',1.2)
plot([0 0],[0 0],'w')
plot(t_fine,prob_error2(:,1),'b','LineWidth',1.2)
plot(t_fine,prob_error3(:,1),'--b','LineWidth',1.2)
hold off
%set(gca,'xtick',[])
yticks([10^(-15),10^(-10),10^(-5),10^(0)])
set(gca,'yscale','log')
ylim([10^(-15) 10^(0)]);
xlabel('$t$','interpreter','latex'); ylabel('$u_1$ error','interpreter','latex');
box on; grid on;
legend('northeast',{'Fine - Parareal',' ','Fine - PPODE (no data)','Fine - PPODE (with data)','Tolerance'},'numcolumns',2)

subplot(2,1,2)
hold on
plot(t_fine,para_error2(:,2),'r','LineWidth',1.2)
plot([0 0],[0 0],'w')
plot(t_fine,prob_error2(:,2),'b','LineWidth',1.2)
plot(t_fine,prob_error3(:,2),'--b','LineWidth',1.2)
hold off
%set(gca,'xtick',[])
yticks([10^(-15),10^(-10),10^(-5),10^(0)])
set(gca,'yscale','log')
ylim([10^(-15) 10^(0)]);
xlabel('$t$','interpreter','latex'); ylabel('$u_2$ error','interpreter','latex');
box on; grid on;




