%% Rossler system
%Here we solve the Rossler system - three coupled ODEs that exhibit chaotic
% behaviour under the parameters given below. To run the 
%code, follow the description and instructions in each section.

%WARNING: some of the code here is designed to run on a HPC system with at least 
%40 compute cores and high RAM storage. There will be a note in that section for
%what to do in this case - note that that results may differ between machines
%and parallel speedup may not be realised. 


%% Solve system (with legacy data)
% This section solves the system using legacy data from a simulation over
% a shorter time interval. First we generate the legacy data, then we solve
% over a longer time interval both with and without the legacy data. 

clear; close all; clc;

% INPUTS 
a = 0.2; b = 0.2; c = 5.7;
f = @(t,u)([-u(2)-u(3);u(1)+(a*u(2));b + u(3)*(u(1)-c)]);    %function handle for ODE
tspan = [0,170];                   %time interval
u0 = [0,-6.78,0.02];               %intial conditions
N = 20;                            %no. of time sub-intervals steps
Ng = 45000;                        %no. of coarse steps
Nf = 2250000;                    %no. of fine steps
epsilon = 10^(-6);                 %error tolerance 
G = 'RK1';                         %coarse solver (see RK.m file)
F = 'RK4';                         %fine solver   (see RK.m file)

% Gaussian Process emulator inputs
kernel = struct('func',@isoSE_cov,'theta',[1,1],'jitter',10*eps);

%solve FHN model using GParareal and parareal (over short interval)
[t,u,err,k,gp,xx,DD] = GParareal(f,tspan,u0,N,Ng,Nf,epsilon,F,G,kernel);
[~,u_p,err_para,k_para] = parareal(f,tspan,u0,N,Ng,Nf,epsilon,F,G);

% now solve over longer interval
tspan2 = [0,tspan(2)*2];               %time interval
N2 = N*2;                                %no. of time sub-intervals steps
Ng2 = Ng*2;                            %no. of coarse steps
Nf2 = Nf*2;                            %no. of fine steps

[t2,u2,err2,k2,gp2,~,~] = GParareal(f,tspan2,u0,N2,Ng2,Nf2,epsilon,F,G,kernel);                       % without legacy data
[~,u2_leg,err2_leg,k2_leg,gp2_leg,~,~] = GParareal(f,tspan2,u0,N2,Ng2,Nf2,epsilon,F,G,kernel,xx,DD);  % with legacy data
[~,u_p2,err_para2,k_para2] = parareal(f,tspan2,u0,N2,Ng2,Nf2,epsilon,F,G);                            % parareal solution

%save('rossler_solutions.mat')

%% Plots: Fiure 4.5(a),(b),(c)
%Here we plot the solutions found in the section above. 

clear; close all; clc;

load('rossler_solutions.mat')

%solve probem for long time interval using fine solver
n = length(u0);
dt = (tspan2(2)-tspan2(1))/Nf2;
t_fine = (tspan2(1):dt:tspan2(end)); 
[~,F_sol2] = RK(t_fine,u0,f,F);         
 


% Fig 1: GParareal/parareal solutions vs. fine

%integrate the ODE using initial values found by GParareal and parareal
fine_trajecs3 = cell(N2,1);
fine_trajecs4 = cell(N2,1);
fine_trajecs5 = cell(N2,1);
parfor i = 1:N2
    [~,temp3] = RK((t2(i):dt:t2(i+1)),u2(i,(n*(k2-1)+1:n*k2)),f,F);
    [~,temp4] = RK((t2(i):dt:t2(i+1)),u2_leg(i,(n*(k2_leg-1)+1:n*k2_leg)),f,F);
    [~,temp5] = RK((t2(i):dt:t2(i+1)),u_p2(i,(n*(k_para2-1)+1:n*k_para2)),f,F);
    if i < N2
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

figure(1)
hold on
plot(F_sol2(:,1),F_sol2(:,2),'color',[0,0,0,0.4])
plot(u2_leg(:,(n*(k2_leg-1)+1)),u2_leg(:,(n*(k2_leg-1)+2)),'ob')
plot(u_p2(:,(n*(k_para2-1)+1)),u_p2(:,(n*(k_para2-1)+2)),'*r')
xlabel('$u_1$','Interpreter','latex'); ylabel('$u_2$','Interpreter','latex');
grid on; box on;
legend({'Fine','GParareal','Parareal'},'Interpreter','latex','location','northeast')
hold off
axis([-10 14 -12 12])

% Fig 2: Max absolute errors over succesive iterations 
err2(cumsum(cumsum(err2~=0)) == 1) = 0; err2(end,end) = eps; errs2 = max(err2,[],1); 
err2_leg(cumsum(cumsum(err2_leg~=0)) == 1) = 0; err2_leg(end,end) = eps; errs2_leg = max(err2_leg,[],1); 
err_para2(cumsum(cumsum(err_para2~=0)) == 1) = 0; err_para2(end,end) = eps; errs_para2 = max(err_para2,[],1); 

figure(2)
hold on
plot((1:k_para2),errs_para2(1:k_para2),'-or','LineWidth',1.2);
plot((1:k2),errs2(1:k2),'-ob','LineWidth',1.2);
plot((1:k2_leg),errs2_leg(1:k2_leg),'color','[0.4660 0.6740 0.1880]','marker','o','LineWidth',1.2);
yline(epsilon,'--k','LineWidth',1.2)
xlabel('$k$','Interpreter','latex'); ylabel('Max. absolute error','Interpreter','latex');
set(gca,'yscale','log')
grid on; box on;
xlim([1,max([k2,k2_leg,k_para2])]); ylim([10^(-7),10^(4)]);
xticks((1:max([k2,k2_leg,k_para2]))); %yticks((0:0.5:5));
legend('northeast',{'Parareal','GParareal (no legacy)','GParareal (legacy)','Tolerance'})
hold off


% SOLUTION ERRORS: FINE vs. GParareal vs. PARAREAL
prob_error2 = abs(F_sol2 - u_GPara2); clear u_prob2; prob_error2(prob_error2==0) = 10^(-40);
prob_error3 = abs(F_sol2 - u_GPara3); clear u_prob3; prob_error3(prob_error3==0) = 10^(-40);
para_error2 = abs(F_sol2 - u_para2); clear u_para2; para_error2(para_error2==0) = 10^(-40);
clear F_sol2;


% Fig 3: Absolute errors of the solutions over time 
figure(3)

h(1) = subplot(3,1,1);
hold on
plot(t_fine,para_error2(:,1),'r','LineWidth',1.2)
plot(t_fine,prob_error2(:,1),'b','LineWidth',1.2)
plot(t_fine,prob_error3(:,1),'color','[0.4660 0.6740 0.1880]','LineWidth',1.2)
hold off
set(gca,'xticklabels',[])
yticks(10.^[-15,-10,-5,0])
set(gca,'yscale','log')
ylim([10^(-15) 10^(0)]);
ylabel('$u_1$ error','interpreter','latex');
box on; grid on;

h(2) = subplot(3,1,2);
hold on
plot(t_fine,para_error2(:,2),'r','LineWidth',1.2)
plot(t_fine,prob_error2(:,2),'b','LineWidth',1.2)
plot(t_fine,prob_error3(:,2),'color','[0.4660 0.6740 0.1880]','LineWidth',1.2)
hold off
set(gca,'xticklabels',[])
yticks(10.^[-15,-10,-5,0])
set(gca,'yscale','log')
ylim([10^(-15) 10^(0)]);
ylabel('$u_2$ error','interpreter','latex');
box on; grid on;


h(3) = subplot(3,1,3);
hold on
plot(t_fine,para_error2(:,3),'r','LineWidth',1.2)
plot(t_fine,prob_error2(:,3),'b','LineWidth',1.2)
plot(t_fine,prob_error3(:,3),'color','[0.4660 0.6740 0.1880]','LineWidth',1.2)
hold off
%set(gca,'xtick',[])
yticks(10.^[-15,-10,-5,0])
set(gca,'yscale','log')
ylim([10^(-15) 10^(0)]);
xlabel('$t$','interpreter','latex'); ylabel('$u_3$ error','interpreter','latex');
box on; grid on;
legend('northeast',{'Fine - Parareal','Fine - GParareal (no legacy)','Fine - GParareal (legacy)'})


%% Parallel speedup 
%Here we calculate runtimes for the problem using up to 40 cores. This 
% section can take a long time to run so do so at your own risk! Skip to
% next section to plot pre-run results. 

clear; close all; clc;

load('rossler_solutions.mat')

% number of different cores to run on 
cores = [1,2,4,8,16,32,40]';
runs = 5;

%storage matrices for timings
GParareal_times = zeros(length(cores),runs+1); GParareal_times(:,1) = cores;
GParareal_legacy_times = zeros(length(cores),runs+1); GParareal_legacy_times(:,1) = cores;
para_times = zeros(length(cores),runs+1);  para_times(:,1) = cores;

for i = length(cores):-1:1
    parpool(cores(i))
    
    for j = 2:runs+1
        T = tic;
        [~,~,~,~,~,~,~] = GParareal(f,tspan2,u0,N2,Ng2,Nf2,epsilon,F,G,kernel);
        s = toc(T);
        GParareal_times(i,j) = s;
        
        T = tic;
        [~,~,~,~,~,~,~] = GParareal(f,tspan2,u0,N2,Ng2,Nf2,epsilon,F,G,kernel,xx,DD);
        s = toc(T);
        GParareal_legacy_times(i,j) = s;
        
        T = tic;
        [~,~,~,~] = parareal(f,tspan2,u0,N2,Ng2,Nf2,epsilon,F,G);
        s = toc(T);
        para_times(i,j) = s;
    end
    
    delete(gcp)
%     save('timings.mat')
end

% fine time mesh
dt2 = (tspan2(2)-tspan2(1))/Nf2;
t_fine2 = (tspan2(1):dt2:tspan2(end));

F_times = zeros(runs,1);
for j = 1:runs
    T = tic;
    [~,~] = RK(t_fine2,u0,f,F);
    s = toc(T);
    F_times(j,1) = s;
end
clear t_fine
% save('timings.mat')



%% Plot Figure 4.5(d)
%Here we plot the results from the section above. 

clear; close all; clc;

load('timings.mat')

%calculate average times for each number of processors
avg_GParareal = median(GParareal_times(:,2:end),2);
avg_legacy_GParareal = median(GParareal_legacy_times(:,2:end),2);
avg_para = median(para_times(:,2:end),2);
fine_time = median(F_times);

%Fig. 1: average run times against processors used (+ corresponding
% speed-up)
figure(3)
hold on
plot(log2(GParareal_times(:,1)),log2(avg_para),'-or','LineWidth',1.2)
plot(log2(GParareal_times(:,1)),log2(avg_GParareal),'-ob','LineWidth',1.2)
plot(log2(GParareal_times(:,1)),log2(avg_legacy_GParareal),'color','#77AC30','marker','o','LineWidth',1.2)
plot(log2([1 64]),log2([fine_time fine_time]),'--k','LineWidth',1.2)
xlabel('Processors','interpreter','latex'); ylabel('$\log_2$(wallclock time)','interpreter','latex');
xticklabels(2.^(0:6))
ylim([8 15.5])
grid on; box on;
legend({'Parareal','GParareal (no legacy)','GParareal (legacy)','Fine Solver'},'interpreter','latex','location','southwest')
hold off

axes('Position',[.50 .58 .38 .31]);
hold on
plot(log2(GParareal_times(:,1)),fine_time./avg_para,'-or','LineWidth',1.2)
plot(log2(GParareal_times(:,1)),fine_time./avg_GParareal,'-ob','LineWidth',1.2)
plot(log2(GParareal_times(:,1)),fine_time./avg_legacy_GParareal,'color','#77AC30','marker','o','LineWidth',1.2)
plot(log2([1 64]),[1 1],'--k','LineWidth',1.2)
xlabel('Processors','interpreter','latex'); ylabel('Speedup','interpreter','latex');
ylim([0 4])
xticks((0:6))
yticks((0:4))
xticklabels(2.^(0:6))
grid on; box on;
hold off

