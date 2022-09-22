%% Double pendulum
%Here we solve the double pendulum system (chaotic). To run the 
%code, follow the description and instructions in each section.

%WARNING: some of the code here is designed to run on a HPC system with at least 
%512 compute cores and high RAM storage. There will be a note in that section for
%what to do in this case - note that that results may differ between machines
%and parallel speedup may not be realised. 

%% Plots: Figure 4.9(a),(b)
% Here we plot solutions using GParareal and parareal, in addition to some
% diagnostics.

clear; close all; clc;

% INPUTS
%function handle for ODE
f = @(t,u)([u(2);...
    (-1/(2 - cos(u(1)-u(3))^2))*((u(2)^2)*cos(u(1)-u(3))*sin(u(1)-u(3)) + (u(4)^2)*sin(u(1)-u(3)) + 2*sin(u(1)) - cos(u(1)-u(3))*sin(u(3)) );...
    u(4);...
    (-1/(2 - cos(u(1)-u(3))^2))*(-2*(u(2)^2)*sin(u(1)-u(3)) - (u(4)^2)*sin(u(1)-u(3))*cos(u(1)-u(3)) - 2*cos(u(1)-u(3))*sin(u(1)) + 2*sin(u(3)) )]);    

tspan = [0,80];                      %time interval
u0 = [2,0,0.5,0];                    %intial conditions (good)
N = 32;                              %no. of time sub-intervals steps
Ng = 3072;                           %no. of coarse steps
Nf = Ng*70;                           %no. of fine steps 
epsilon = 10^(-6);                   %error tolerance 
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
[t,u,err,k,gp_kers,xx,DD,T] = GParareal(f,tspan,u0,N,Ng,Nf,epsilon,F,G,kernel);
[~,u_p,err_para,k_para,slices] = parareal(f,tspan,u0,N,Ng,Nf,epsilon,F,G);


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
plot([tspan(1), tspan(2)],[pi pi],'--k')
plot([tspan(1), tspan(2)],[-pi -pi],'--k')
hold off
xlim(tspan);
ylim([-pi-0.5 pi+0.5]); yticks((-pi:(pi/2):pi)); yticklabels({'-\pi','','0','','\pi'})
set(gca,'xticklabels',[])
ylabel('$u_1(t)$','interpreter','latex');
box on; grid on;

subplot(2,1,2)
hold on
plot(t_fine,F_sol(:,3),'k')
plot(t_fine((1:(Nf/N):end)),u_prob((1:(Nf/N):end),3),'ob')
plot(t_fine((1:(Nf/N):end)),u_para((1:(Nf/N):end),3),'*r')
plot([tspan(1), tspan(2)],[pi pi],'--k')
plot([tspan(1), tspan(2)],[-pi -pi],'--k')
plot([tspan(1), tspan(2)],[-3*pi -3*pi],'--k')
plot([tspan(1), tspan(2)],[-5*pi -5*pi],'--k')
plot([tspan(1), tspan(2)],[-7*pi -7*pi],'--k')
hold off
xlim(tspan);
ylim([-7*pi-1.5 pi+1.5]); yticks((-7*pi:pi:pi)); yticklabels({'-7\pi','','-5\pi','','-3\pi','','-\pi','','\pi'})
xlabel('$t$','interpreter','latex'); ylabel('$u_2(t)$','interpreter','latex');
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
yline(epsilon,'--k')
xlabel('$k$ (Iterations)','Interpreter','latex'); ylabel('Max. Absolute Error','Interpreter','latex');
set(gca,'yscale','log')
grid on; box on;
xlim([1,max([k,k_para,5])]); ylim([10^(-8),10^(4)]);
xticks((1:max([k,k_para]))); %yticks((0:0.5:5));
legend('northeast',{'Parareal','GParareal','Tolerance'})
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



%% Solve system for various ICs (heat maps)
% This section runs GParareal and parareal over many initial values to compare
% convergence rates in two heat maps. WARNING: takes a while to run. Skip
% to next section to plot pre-run results. 

clear; close all; clc;

% INPUTS
f = @(t,u)([u(2);...
    (-1/(2 - cos(u(1)-u(3))^2))*((u(2)^2)*cos(u(1)-u(3))*sin(u(1)-u(3)) + (u(4)^2)*sin(u(1)-u(3)) + 2*sin(u(1)) - cos(u(1)-u(3))*sin(u(3)) );...
    u(4);...
    (-1/(2 - cos(u(1)-u(3))^2))*(-2*(u(2)^2)*sin(u(1)-u(3)) - (u(4)^2)*sin(u(1)-u(3))*cos(u(1)-u(3)) - 2*cos(u(1)-u(3))*sin(u(1)) + 2*sin(u(3)) )]);    

tspan = [0,80];                      %time interval
N = 32;                             %no. of time sub-intervals steps
Ng = 3072;                            %no. of coarse steps
Nf = Ng*7000000;                      %no. of fine steps
epsilon = 10^(-6);                    %error tolerance 
G = 'RK1';                            %coarse solver
F = 'RK8';                            %fine solver

%fine solve
dt = (tspan(2)-tspan(1))/Nf;            % size of fine time step
t_fine = (tspan(1):dt:tspan(end));      % fine time step mesh


kernel = struct('func',@isoSE_cov,'theta',[1,1],'jitter',1e-12);

% set of ICs to run
u1 = linspace(-2.5,2.5,11);
u2 = linspace(2.5,-2.5,11);

%store convergence rates for each simulation
k_GParareal = zeros(length(u2),length(u1));
k_para = k_GParareal;
turn = k_GParareal;

% find each convergence rate (GParareal)
for i = 1:length(u2)
    for j = 1:length(u1)
        [~,u,~,k,~,~,~,~] = GParareal(f,tspan,[u1(j),0,u2(i),0],N,Ng,Nf,epsilon,F,G,kernel);
        k_GParareal(i,j) = k;
        if sum(([u(:,end-3),u(:,end-1)] > pi),'all') ~= 0
            turn(i,j) = 1;
        end
    end
end
%save('DP_heatmap.mat')


% find each convergence rate (parareal)
for i = 1:length(u2)
    for j = 1:length(u1)
        [~,~,~,k] = parareal(f,tspan,[u1(j),0,u2(i),0],N,Ng,Nf,epsilon,F,G);
        k_para(i,j) = k;
    end
end
%save('DP_heatmap.mat')

%% Plots: Figure 4.10(a),(b)
% Plot results from section above. 

clear; close all; clc

load('DP_heatmap.mat')

%fix colormap
B = jet;

%plot the heatmap (parareal)
figure(1)

h2 = heatmap(u1,u2,k_para);
%h2.ColorLimits = [10,15];
colormap(B)
colorbar off
h2.XLabel = '$u_1(t=0)$';
h2.YLabel = '$u_2(t=0)$';
h2.NodeChildren(3).XAxis.Label.Interpreter = 'latex';
h2.NodeChildren(3).YAxis.Label.Interpreter = 'latex';



%plot the heatmap (GParareal)
figure(2)
h1 = heatmap(u1,u2,k_GParareal);
%h1.ColorLimits = [2,6];
colormap(B)
colorbar off
h1.XLabel = '$u_1(t=0)$';
h1.YLabel = '$u_2(t=0)$';
h1.NodeChildren(3).XAxis.Label.Interpreter = 'latex';
h1.NodeChildren(3).YAxis.Label.Interpreter = 'latex';


%% Plots: Figure 4.11(a),(b) and Table 4.2
% Here we display the results given in Table 4.2 and plotted in Figure 4.11.
% These results are pulled from saved .mat files generated by running
% GParareal and parareal on 512 cores (these are stored in 'Test 2' folder).

% To run this section, you need to add the 'Seclected Folders and
% Subfolders' of 'Test 2' folder to your path. Do this by right-clicking 'Test 2'.

% If interested in running the code for these results, please contact the
% lead author. 

clear; close all; clc;

res = cell(8,17);
res(1,:) = {'J','kgpara','kpara','Tg','Tf','Tgp','Tser','Tpara','Tgpara','Spara','Sgpara','Tpara','Tgpara','Spara','Sgpara','Epara','Egpara'};
% res(2,6:8) = {'Actual'};
res(2,12:15) = {'Theory'};
res(3:end,1) = {'32','64','128','256','512','1024'};


load('GP_DP_N32.mat')
i = 3;
res{i,2} = k;

res{i,4} = T{1};
res{i,5} = T{2};
res{i,6} = sum(T{3},'all'); %max(sum(T{3},1));
res{i,7} = N*T{2};
res{i,8} = 130803.82;          % Tpara (manual)
res{i,9} = 121019.99;          % Tgpara (manual)
res{i,10} = res{i,7}/res{i,8};
res{i,11} = res{i,7}/res{i,9};

res{i,13} = k*T{2} + (k+1)*(N - (k/2))*T{1} + res{i,6};
res{i,15} = 1/((k/N) + (k+1)*(1 - (k/(2*N)))*(T{1}/T{2}) + (1/(N*T{2}))*res{i,6});
res{i,17} = res{i,11}/res{i,15};

load('P_DP_N32.mat')
res{i,3} = k_para;
res{i,12} = k_para*T{2} + (k_para+1)*(N - (k_para/2))*T{1};
res{i,14} = 1/((k_para/N) + (k_para+1)*(1 - (k_para/(2*N)))*(T{1}/T{2}));
res{i,16} = res{i,10}/res{i,14};


load('GP_DP_N64.mat')
i = 4;
res{i,2} = k;

res{i,4} = T{1};
res{i,5} = T{2};
res{i,6} = sum(T{3},'all'); %max(sum(T{3},1));
res{i,7} = N*T{2};
res{i,8} = 62875.16;          % Tpara (manual)
res{i,9} = 70023.60;          % Tgpara (manual)
res{i,10} = res{i,7}/res{i,8};
res{i,11} = res{i,7}/res{i,9};

res{i,13} = k*T{2} + (k+1)*(N - (k/2))*T{1} + res{i,6};
res{i,15} = 1/((k/N) + (k+1)*(1 - (k/(2*N)))*(T{1}/T{2}) + (1/(N*T{2}))*res{i,6});
res{i,17} = res{i,11}/res{i,15};

load('P_DP_N64.mat')
res{i,3} = k_para;
res{i,12} = k_para*T{2} + (k_para+1)*(N - (k_para/2))*T{1};
res{i,14} = 1/((k_para/N) + (k_para+1)*(1 - (k_para/(2*N)))*(T{1}/T{2}));
res{i,16} = res{i,10}/res{i,14};


load('GP_DP_N128.mat')
i = 5;
res{i,2} = k;

res{i,4} = T{1};
res{i,5} = T{2};
res{i,6} = sum(T{3},'all'); %max(sum(T{3},1));
res{i,7} = N*T{2};
res{i,8} = 38523.35;          % Tpara (manual)
res{i,9} = 35587.74;          % Tgpara (manual)
res{i,10} = res{i,7}/res{i,8};
res{i,11} = res{i,7}/res{i,9};

res{i,13} = k*T{2} + (k+1)*(N - (k/2))*T{1} + res{i,6};
res{i,15} = 1/((k/N) + (k+1)*(1 - (k/(2*N)))*(T{1}/T{2}) + (1/(N*T{2}))*res{i,6});
res{i,17} = res{i,11}/res{i,15};

load('P_DP_N128.mat')
res{i,3} = k_para;
res{i,12} = k_para*T{2} + (k_para+1)*(N - (k_para/2))*T{1};
res{i,14} = 1/((k_para/N) + (k_para+1)*(1 - (k_para/(2*N)))*(T{1}/T{2}));
res{i,16} = res{i,10}/res{i,14};



load('GP_DP_N256.mat')
i = 6;
res{i,2} = k;

res{i,4} = T{1};
res{i,5} = T{2};
res{i,6} = sum(T{3},'all'); %max(sum(T{3},1));
res{i,7} = N*T{2};
res{i,8} = 16619.54;          % Tpara (manual)
res{i,9} = 20447.92;          % Tgpara (manual)
res{i,10} = res{i,7}/res{i,8};
res{i,11} = res{i,7}/res{i,9};

res{i,13} = k*T{2} + (k+1)*(N - (k/2))*T{1} + res{i,6};
res{i,15} = 1/((k/N) + (k+1)*(1 - (k/(2*N)))*(T{1}/T{2}) + (1/(N*T{2}))*res{i,6});
res{i,17} = res{i,11}/res{i,15};

load('P_DP_N256.mat')
res{i,3} = k_para;
res{i,12} = k_para*T{2} + (k_para+1)*(N - (k_para/2))*T{1};
res{i,14} = 1/((k_para/N) + (k_para+1)*(1 - (k_para/(2*N)))*(T{1}/T{2}));
res{i,16} = res{i,10}/res{i,14};




load('GP_DP_N512.mat')
i = 7;
res{i,2} = k;

res{i,4} = T{1};
res{i,5} = T{2};
res{i,6} = sum(T{3},'all'); %max(sum(T{3},1));
res{i,7} = N*T{2};
res{i,8} = 7581.04;          % Tpara (manual)
res{i,9} = 22522.73;          % Tgpara (manual)
res{i,10} = res{i,7}/res{i,8};
res{i,11} = res{i,7}/res{i,9};

res{i,13} = k*T{2} + (k+1)*(N - (k/2))*T{1} + res{i,6};
res{i,15} = 1/((k/N) + (k+1)*(1 - (k/(2*N)))*(T{1}/T{2}) + (1/(N*T{2}))*res{i,6});
res{i,17} = res{i,11}/res{i,15};

load('P_DP_N512.mat')
res{i,3} = k_para;
res{i,12} = k_para*T{2} + (k_para+1)*(N - (k_para/2))*T{1};
res{i,14} = 1/((k_para/N) + (k_para+1)*(1 - (k_para/(2*N)))*(T{1}/T{2}));
res{i,16} = res{i,10}/res{i,14};







res


% PLOTS

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
ylim([12 18])
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
ylim([0 30]); yticks((0:5:25))
grid on; box on;
legend({'Parareal (theory)','Parareal (numerical)','GParareal (theory)','GParareal (numerical)','Fine Solver'},'interpreter','latex','location','northwest')
hold off




