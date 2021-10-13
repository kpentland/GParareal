%% Test Scripts
% To test the algorithm, click any of the sections below and "run section" (above). 


%% Logistic Equation (1D)
%Here we solve the Logistic equation with growth rate r = 1. Plots are described below.  

clear; close all; clc;

% INPUTS
u0 = 5;                            %initial condition for ODE
f = @(t,u)( 1*u*(1-u) );           %the ODE to be solved
auto = 1;                          %is the system autonomous (yes=1, no=0)
tspan = [0,15];                    %time interval of integration
epsilon = 10^(-8);                 %error tolerance
N = 20;                            %no. of time sub-intervals (processors)
Ng = 80;                           %coarse time steps
Nf = 10000;                        %fine time steps
G = 'RK1';                         %coarse solver
F = 'RK4';                         %fine solver

% GP prior inputs
K = gpcf_sexp('lengthScale',1,'magnSigma2',1);  %square exponential covariance kernel
like = lik_gaussian();                          %Gaussian likelihood with prior scale parameter


%%PLOTS

%fine time step and mesh
dt = (tspan(2)-tspan(1))/Nf;
t_fine = (tspan(1):dt:tspan(end));


% Fig 1: PPODE, fine, parareal solutions vs. analytical
[t,u,err,k,gp,xx,DD] = PPODE(f,auto,tspan,u0,N,Ng,Nf,epsilon,F,G,K,like);     % PPODE solution
[~,F_sol] = RK(t_fine,u0,f,F);                                           % fine solution
[~,u_p,err_para,k_para] = parareal(f,tspan,u0,N,Ng,Nf,epsilon,F,G);      % parareal solution
u_true = (u0*exp(1*t_fine)./(1 - u0 + u0*exp(1*t_fine)))';               % analytical solution

% display optimised GP hyperparameters
fprintf('Optimised GP hyperparameters: \n')
[w,s] = gp_pak(gp{:}); 
disp(s), disp(w)

%integrate the ODE using initial values found by PPODE and parareal
fine_trajecs1 = cell(N,1);
fine_trajecs2 = cell(N,1);
for i = 1:N
    [~,temp1] = RK((t(i):dt:t(i+1)),u(i,end),f,F);
    [~,temp2] = RK((t(i):dt:t(i+1)),u_p(i,end),f,F);
    if i < N
    fine_trajecs1{i,1} = temp1(1:end-1,:);                 
    fine_trajecs2{i,1} = temp2(1:end-1,:);                 
    else
    fine_trajecs1{i,1} = temp1;                 
    fine_trajecs2{i,1} = temp2;                 
    end
end
u_prob = vertcat(fine_trajecs1{:,1});            % PPODE high-resolution solution
u_para = vertcat(fine_trajecs2{:,1});            % parareal high-resolution solution


%plot the solutions and the associated errors
figure(1)
hold on
plot(t_fine,u_true,'k','LineWidth',2);
plot(t_fine(1:(Nf/N):end),u_para(1:(Nf/N):end),'*r')
plot(t_fine(1:(Nf/N):end),u_prob((1:(Nf/N):end),end),'ob')
xlabel('$t$','Interpreter','latex'); ylabel('$u(t)$','Interpreter','latex');
grid on; box on;
axis([0 15 1 5])
xticks((0:2.5:15)); yticks((0:0.5:5));
legend('northeast',{'Analytical','Parareal','PPODE'},'Interpreter','latex')
hold off

axes('Position',[.375 .26 .505 .47])
hold on
plot(t_fine,abs(u_true - F_sol),'k','LineWidth',1.2)
plot(t_fine,abs(u_true - u_prob),'--b','LineWidth',1.2)
plot(t_fine,abs(u_true - u_para),'--r','LineWidth',1.2)
legend({'Analytical - Fine','Analytical - PPODE','Analytical - Parareal'},'interpreter','latex','location','northeast')
xlabel('$t$','interpreter','latex'); ylabel('Error','interpreter','latex');
set(gca,'yscale','log')
ylim([10^(-15) 10^(-5)]);
%xticks((0:2:10)); yticks([10^(-14),10^(-12),10^(-10),10^(-8)]);
box on; grid on;



%Fig 2: plot the GP regression over (F-G)(x)

% test input initial conditions (where to evaluate the GP)
x_star = sort([linspace(min(u,[],'all'),max(u,[],'all'),501)';xx]);

% calculate the GP posterior 
[post_mean,post_cov] = gp_pred(gp{:},xx,DD,x_star);

%calculate the confidence interval (2 standard devs)
stan_dev = sqrt(abs(post_cov));
ub = post_mean + 2*stan_dev;     lb = post_mean - 2*stan_dev;

figure(2)
hold on
fill([x_star', fliplr(x_star')],[ub', flipud(lb)'], 1,'facecolor','b', 'facealpha', 0.3);  %fill in confidence interval
plot(x_star,post_mean,'k','linewidth',1.5)     %plots posterior mean
scatter(xx,DD,250,'.r')                        %plots the observations
ylim([-1,1])
legend({'$\pm$ two standard deviations','Mean','Observations/data'},'interpreter','latex','location','southwest')
xlabel('Initial value $x$','interpreter','latex'); ylabel('$(\mathcal{F}-\mathcal{G})(x)$','interpreter','latex');
grid on; box on;
help

%Fig 3: Convergence plot (PPODE vs. parareal)

%calculate maximum error during each iteration (for each algorithm)
err(cumsum(cumsum(err~=0)) == 1) = 0; err(end,end) = eps; errs = max(err,[],1); 
err_para(cumsum(cumsum(err_para~=0)) == 1) = 0; err_para(end,end) = eps; errs_para = max(err_para,[],1); 

figure(3)
hold on
plot((1:k_para),errs_para(1:k_para),'-ok','LineWidth',1.2);
plot((1:k),errs(1:k),'-ob','LineWidth',1.2);
yline(epsilon,'--k')
xlabel('Iteration $k$','Interpreter','latex'); ylabel('Max. Absolute Error','Interpreter','latex');
set(gca,'yscale','log')
grid on; box on;
xlim([1,max(k,k_para)]); ylim([10^(-10),1]);
xticks((1:max(k,k_para))); %yticks((0:0.5:5));
legend('northeast',{'Parareal','PPODE','Tolerance'})
hold off

%% Non-autonomous Equation (1D)
%Here we solve a non-autonomous ODE. Plots are described below.  

clear; close all; clc;

% INPUTS
f = @(t,u)(u*cos(u*t)+cos(t));     %the ODE to be solved for parareal
tspan = [0,15];                    %time interval of integration
u0 = 1;                            %initial condition for ODE system + initial time of integration
auto = 0;                          %is the system autonomous (yes=1, no=0)
epsilon = 10^(-8);                 %error tolerance
N = 25;                            %no. of time sub-intervals (processors)
Ng = 250;                           %coarse time steps
Nf = Ng*500;                       %fine time steps
G = 'RK1';                         %coarse solver
F = 'RK4';                         %fine solver

% Gaussian Process prior inputs
K = gpcf_sexp('lengthScale',1, 'magnSigma2',1);  %square exponential kernel
like = lik_gaussian('sigma2', 1);                %Gaussian likelihood with prior scale parameter

%%PLOTS

%fine time step and mesh
n = length(u0);
dt = (tspan(2)-tspan(1))/Nf;
t_fine = (tspan(1):dt:tspan(end));


% Fig 1: PPODE, fine, parareal solutions vs. analytical
[t,u,err,k,gp,xx,DD] = PPODE(f,auto,tspan,u0,N,Ng,Nf,epsilon,F,G,K,like);     % PPODE solution
[~,F_sol] = RK(t_fine,u0,f,F);                                                % fine solution
[~,u_p,err_para,k_para] = parareal(f,tspan,u0,N,Ng,Nf,epsilon,F,G);           % parareal solution

% display optimised GP hyperparameters
fprintf('Optimised GP hyperparameters: \n')
[w,s] = gp_pak(gp{:}); 
disp(s), disp(w)

% use the ICs found by each algorithm to integrate in parallel
np = 1;
fine_trajecs1 = cell(N,1);
fine_trajecs2 = cell(N,1);
for i = 1:N
    [~,temp1] = RK((t(i):dt:t(i+1)),u(i,(n*(k-1)+1:n*k)),f,F);
    [~,temp2] = RK((t(i):dt:t(i+1)),u_p(i,(np*(k_para-1)+1:np*k_para)),f,F);
    if i < N
        fine_trajecs1{i,1} = temp1(1:end-1,:);
        fine_trajecs2{i,1} = temp2(1:end-1,:);
    else
        fine_trajecs1{i,1} = temp1;
        fine_trajecs2{i,1} = temp2;
    end
end
u_prob = vertcat(fine_trajecs1{:,1});            % PPODE high-resolution solution
u_para = vertcat(fine_trajecs2{:,1});            % parareal high-resolution solution


%plot the solutions
figure(1)
hold on
plot(t_fine,F_sol(:,1),'k')
plot(t_fine(1:(Nf/N):end),u_para((1:(Nf/N):end),1),'*r')
plot(t_fine(1:(Nf/N):end),u_prob((1:(Nf/N):end),1),'ob')
xlabel('$t$','Interpreter','latex'); ylabel('$u$','Interpreter','latex');
grid on; box on;
legend('northeast',{'Fine','Parareal','PPODE'},'Interpreter','latex')
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
xlabel('Iteration $k$','Interpreter','latex'); ylabel('Max. Absolute Error','Interpreter','latex');
set(gca,'yscale','log')
grid on; box on;
xlim([1,max(k,k_para)]); ylim([10^(-10),1]);
xticks((1:max(k,k_para))); %yticks((0:0.5:5));
legend('northeast',{'Parareal','PPODE','Tolerance'})
hold off


%Fig 3: Error plot (PPODE/parareal vs. fine solution)

%calculate absolute error at each time step (for each algorithm)
prob_error1 = abs(F_sol - u_prob); prob_error1(prob_error1==0) = 10^(-40);
para_error1 = abs(F_sol - u_para); para_error1(para_error1==0) = 10^(-40);

figure(3)
hold on
plot(t_fine,para_error1,'r','LineWidth',1.2)
plot(t_fine,prob_error1,'b','LineWidth',1.2)
hold off
set(gca,'yscale','log')
xlim(tspan)
ylim([10^(-15) 10^(0)]);
xlabel('$t$','interpreter','latex'); ylabel('$u_1$ error','interpreter','latex');
box on; grid on;
legend({'Fine - Parareal','Fine - PPODE'},'interpreter','latex','location','northwest')



%Fig 4: mean of the GP regression over (F-G)(x)

% Choose test points on a dense grid
ns = 5625;
xs1 = linspace(tspan(1),tspan(2),sqrt(ns));
xs2 = linspace(min(u_prob(:,1)),max(u_prob(:,1)),sqrt(ns));
[xs1,xs2] = meshgrid(xs1,xs2); 

[means,~] = gp_pred(gp{1},xx,DD,[xs1(:),xs2(:)]);

% Plot the prediction and data
figure(4)
clf
mesh(xs1, xs2, reshape(means,sqrt(ns),sqrt(ns)));
hold on
plot3(xx(:,1),xx(:,2),DD(:,1),'.r')
axis on;
xlabel('$x_1$ (time)','interpreter','latex'); ylabel('$x_2$ (phase space)','interpreter','latex');
zlabel('$(\mathcal{F}-\mathcal{G}) (x_1,x_2)$','interpreter','latex');


%% Non-autonomous system (2D)
%Here we solve a non-autonomous ODE system of two equations. Plots are described below.  

clear; close all; clc;

% INPUTS
f = @(t,u)([-u(2) + u(1)*(t/100 - u(1)^2- u(2)^2);u(1) + u(2)*(t/100 - u(1)^2- u(2)^2)]);  %the ODE to be solved for parareal
tspan = [0,100];                   %time interval of integration
u0 = [0.1,0.1];                    %initial condition for ODE system + initial time of integration
auto = 0;                          %is the system autonomous (yes=1, no=0)
epsilon = 10^(-8);                 %error tolerance
N = 25;                            %no. of time sub-intervals (processors)
Ng = 250;                          %coarse time steps
Nf = Ng*500;                       %fine time steps
G = 'RK1';                         %coarse solver
F = 'RK4';                         %fine solver

% Gaussian Process prior inputs
K = gpcf_sexp('lengthScale',1, 'magnSigma2',1);  %square exponential kernel
like = lik_gaussian('sigma2', 1);                %Gaussian likelihood with prior scale parameter



%%PLOTS

%fine time step and mesh
n = length(u0);
dt = (tspan(2)-tspan(1))/Nf;
t_fine = (tspan(1):dt:tspan(end));


% Fig 1/2: PPODE, fine, parareal solutions vs. analytical
[t,u,err,k,gp,xx,DD] = PPODE(f,auto,tspan,u0,N,Ng,Nf,epsilon,F,G,K,like);     % PPODE solution
[~,F_sol] = RK(t_fine,u0,f,F);                                                % fine solution
[~,u_p,err_para,k_para] = parareal(f,tspan,u0,N,Ng,Nf,epsilon,F,G);           % parareal solution


% display optimised GP hyperparameters
fprintf('Optimised GP hyperparameters - output one: \n')
[w1,s1] = gp_pak(gp{1});  disp(s1), disp(w1)
fprintf('Optimised GP hyperparameters - output two: \n')
[w2,s2] = gp_pak(gp{2});  disp(s2), disp(w2)

% use the ICs found by each algorithm to integrate in parallel
np = n;
fine_trajecs1 = cell(N,1);
fine_trajecs2 = cell(N,1);
for i = 1:N
    [~,temp1] = RK((t(i):dt:t(i+1)),u(i,(n*(k-1)+1:n*k)),f,F);
    [~,temp2] = RK((t(i):dt:t(i+1)),u_p(i,(np*(k_para-1)+1:np*k_para)),f,F);
    if i < N
        fine_trajecs1{i,1} = temp1(1:end-1,:);
        fine_trajecs2{i,1} = temp2(1:end-1,:);
    else
        fine_trajecs1{i,1} = temp1;
        fine_trajecs2{i,1} = temp2;
    end
end
u_prob = vertcat(fine_trajecs1{:,1});            % PPODE high-resolution solution
u_para = vertcat(fine_trajecs2{:,1});            % parareal high-resolution solution


%plot the solutions

figure(1)

subplot(2,1,1)
hold on
plot(t_fine,F_sol(:,1),'k')
plot(t_fine((1:(Nf/N):end)),u_prob((1:(Nf/N):end),1),'ob')
plot(t_fine((1:(Nf/N):end)),u_para((1:(Nf/N):end),1),'*r')
hold off
xlim([0 100]); ylim([-1.5 1.5])
xlabel('$t$','interpreter','latex'); ylabel('$u_1(t)$','interpreter','latex');
box on; grid on;

subplot(2,1,2)
hold on
plot(t_fine,F_sol(:,2),'k')
plot(t_fine((1:(Nf/N):end)),u_prob((1:(Nf/N):end),2),'ob')
plot(t_fine((1:(Nf/N):end)),u_para((1:(Nf/N):end),2),'*r')
hold off
xlim([0 100]); ylim([-1.5 1.5])
xlabel('$t$','interpreter','latex'); ylabel('$u_2(t)$','interpreter','latex');
box on; grid on;

legend({'Fine','PPODE','Parareal'},'Interpreter','latex','numcolumns',3,'location','southwest')
hold off

%plot the solutions in 3D space
figure(2)
hold on
plot3(t_fine,F_sol(:,1),F_sol(:,2),'k')
plot3(t,u_prob((1:(Nf/N):end),1),u_prob((1:(Nf/N):end),2),'ob')
plot3(t,u_para((1:(Nf/N):end),1),u_para((1:(Nf/N):end),2),'*r')
xlabel('Time $t$','Interpreter','latex'); ylabel('$u_1$','Interpreter','latex'); zlabel('$u_2$','Interpreter','latex');
grid on; box on;
view(-20,20)
legend({'Fine','PPODE','Parareal'},'Interpreter','latex','numcolumns',3,'location','northwest')
hold off


%Fig 3: Convergence plot (PPODE vs. parareal)

%calculate maximum error during each iteration (for each algorithm)
err(cumsum(cumsum(err~=0)) == 1) = 0; err(end,end) = eps; errs = max(err,[],1); 
err_para(cumsum(cumsum(err_para~=0)) == 1) = 0; err_para(end,end) = eps; errs_para = max(err_para,[],1); 

figure(3)
hold on
plot((1:k_para),errs_para(1:k_para),'-ok','LineWidth',1.2);
plot((1:k),errs(1:k),'-ob','LineWidth',1.2);
yline(epsilon,'--k')
xlabel('Iteration $k$','Interpreter','latex'); ylabel('Max. Absolute Error','Interpreter','latex');
set(gca,'yscale','log')
grid on; box on;
xlim([1,max(k,k_para)]); ylim([10^(-10),10^(2)]);
xticks((1:max(k,k_para))); %yticks((0:0.5:5));
legend('northeast',{'Parareal','PPODE','Tolerance'})
hold off

%Fig 4: Error plot (PPODE/parareal vs. fine solution)

%calculate absolute error at each time step (for each algorithm)
prob_error1 = abs(F_sol(:,1) - u_prob(:,1)); prob_error1(prob_error1==0) = 10^(-40);
para_error1 = abs(F_sol(:,1) - u_para(:,1)); para_error1(para_error1==0) = 10^(-40);

prob_error2 = abs(F_sol(:,2) - u_prob(:,2)); prob_error2(prob_error2==0) = 10^(-40);
para_error2 = abs(F_sol(:,2) - u_para(:,2)); para_error2(para_error2==0) = 10^(-40);

figure(4)
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
legend({'Fine - Parareal','Fine - PPODE'},'interpreter','latex','location','northwest')

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




%% Brusselator system
%Here we solve the Brusselator system - a pair of coupled stiff ODEs. Plots are described below.  

clear; close all; clc;

% INPUTS
u0 = [1,3.07];                     %initial conditions for ODEs
f = @(t,u)([1 + (u(1)^2)*u(2) - (3+1)*u(1); 3*u(1) - (u(1)^2)*u(2)]);           %the ODEs to be solved
auto = 1;                          %is the system autonomous (yes=1, no=0)
tspan = [0,15.3];                  %time interval of integration
epsilon = 10^(-8);                 %error tolerance
N = 25;                            %no. of time sub-intervals (processors)
Ng = 100;                          %coarse time steps
Nf = Ng*500;                       %fine time steps
G = 'RK1';                         %coarse solver
F = 'RK4';                         %fine solver

% Gaussian Process prior inputs
K = gpcf_sexp('lengthScale',1, 'magnSigma2',1);      %square exponential kernel (isotropic)
like = lik_gaussian();                               %Gaussian likelihood with prior scale parameter

%fine time step and mesh (n = no. of dimensions)
n = length(u0);
dt = (tspan(2)-tspan(1))/Nf;
t_fine = (tspan(1):dt:tspan(end));


%solve problem with original IC
[t,u,err,k,gp,xx,DD] = PPODE(f,auto,tspan,u0,N,Ng,Nf,epsilon,F,G,K,like);     % PPODE solution
[~,F_sol] = RK(t_fine,u0,f,F);                                                % fine solution
[~,u_p,err_para,k_para] = parareal(f,tspan,u0,N,Ng,Nf,epsilon,F,G);           % parareal solution


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
u_prob = vertcat(fine_trajecs1{:,1});            % PPODE high-resolution solution
u_para = vertcat(fine_trajecs2{:,1});            % parareal high-resolution solution

figure(1)
hold on
plot(F_sol(:,1),F_sol(:,2),'k')
plot(u_para((1:(Nf/N):end),1),u_para((1:(Nf/N):end),2),'*r')
plot(u_prob((1:(Nf/N):end),1),u_prob((1:(Nf/N):end),2),'ob')
xlabel('$u_1$','Interpreter','latex'); ylabel('$u_2$','Interpreter','latex');
grid on; box on;
axis([0 4 0 5])
xticks((0:0.5:4)); yticks((0:0.5:5));
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
legend({'Fine - Parareal','Fine - PPODE'},'interpreter','latex','location','northwest')

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
xs1 = linspace(min(xx(:,1)),max(xx(:,1)),sqrt(ns));
xs2 = linspace(min(xx(:,2)),max(xx(:,2)),sqrt(ns));
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
axis([min(xx(:,1)) max(xx(:,1)) min(xx(:,2)) max(xx(:,2))])
axis on;
xlabel('$x_1$','interpreter','latex'); ylabel('$x_2$','interpreter','latex');
zlabel('$(\mathcal{F}-\mathcal{G})_1 (x_1,x_2)$','interpreter','latex');
title('Output one');

figure(5)
clf
mesh(xs1, xs2, reshape(indep_means{2},sqrt(ns),sqrt(ns)));
hold on
plot3(xx(:,1),xx(:,2),DD(:,2),'.r')
axis([min(xx(:,1)) max(xx(:,1)) min(xx(:,2)) max(xx(:,2))])
axis on;
xlabel('$x_1$','interpreter','latex'); ylabel('$x_2$','interpreter','latex');
zlabel('$(\mathcal{F}-\mathcal{G})_2 (x_1,x_2)$','interpreter','latex');
title('Output two');


%% FitzHugh-Nagumo system
%Here we solve the FHN system - another pair of coupled stiff ODEs. Plots are described below.  

clear; close all; clc;

% INPUTS
u0 = [-1,1];                     %initial condition for ODE
f = @(t,u)([3*(u(1) - ((u(1)^3)/3) + u(2) ); -(1/3)*(u(1) - 0.2 + 0.2*u(2))]);    %the ODE to be solved
auto = 1;                          %is the system autonomous (yes=1, no=0)
tspan = [0,32];                    %time interval of integration
epsilon = 10^(-8);                 %error tolerance
N = 32;                            %no. of time sub-intervals (processors)
Ng = 128;                          %coarse time steps
Nf = Ng*500;                       %fine time steps
G = 'RK2';                         %coarse solver
F = 'RK4';                         %fine solver

% Gaussian Process prior inputs
K = gpcf_sexp('lengthScale',1, 'magnSigma2',1);      %square exponential kernel (isotropic)
like = lik_gaussian();                               %Gaussian likelihood with prior scale parameter

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
xlim([0 32]); ylim([-3 3])
xlabel('$t$','interpreter','latex'); ylabel('$u_1(t)$','interpreter','latex');
box on; grid on;

subplot(2,1,2)
hold on
plot(t_fine,F_sol(:,2),'k')
plot(t_fine((1:(Nf/N):end)),u_prob((1:(Nf/N):end),2),'ob')
plot(t_fine((1:(Nf/N):end)),u_para((1:(Nf/N):end),2),'*r')
hold off
xlim([0 32]); ylim([-1.5 1.5])
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
legend({'Fine - Parareal','Fine - PPODE'},'interpreter','latex','location','northwest')

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
xlabel('$x_1$','interpreter','latex'); ylabel('$x_2$','interpreter','latex');
zlabel('$(\mathcal{F}-\mathcal{G})_1 (x_1,x_2)$','interpreter','latex');
title('Output one');

figure(5)
clf
mesh(xs1, xs2, reshape(indep_means{2},sqrt(ns),sqrt(ns)));
hold on
plot3(xx(:,1),xx(:,2),DD(:,2),'.r')
axis([-3 3 -1.5 1.5 -0.03 0.03])
axis on;
xlabel('$x_1$','interpreter','latex'); ylabel('$x_2$','interpreter','latex');
zlabel('$(\mathcal{F}-\mathcal{G})_2 (x_1,x_2)$','interpreter','latex');
title('Output two');


%% Lorenz63 system
%Here we solve the Lorenz63 system in the chaotic parameter regime. Plots are described below.  

clear; close all; clc;

% INPUTS
f = @(t,u)([10*(u(2) - u(1)); 28*u(1) - u(2) - u(1)*u(3); u(1)*u(2) - (8/3)*u(3)]);    %function handle for ODE
auto = 1;                              %is the system autonomous (yes=1, no=0)
tspan = [0,18];                        %time interval
u0 = [-15,-15,20];                     %intial conditions
N = 40;                                %no. of time sub-intervals steps
Ng = 1000;                             %no. of coarse steps
Nf = Ng*500;                           %no. of fine steps
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


%plot the solutions
figure(1)
hold on
plot3(F_sol(:,1),F_sol(:,2),F_sol(:,3),'k')
plot3(u_prob((1:(Nf/N):end),1),u_prob((1:(Nf/N):end),2),u_prob((1:(Nf/N):end),3),'ob')
plot3(u_para((1:(Nf/N):end),1),u_para((1:(Nf/N):end),2),u_para((1:(Nf/N):end),3),'*r')
xlabel('$u_1$','Interpreter','latex'); ylabel('$u_2$','Interpreter','latex'); zlabel('$u_3$','Interpreter','latex');
grid on; box on;
view(50,15)
legend({'Fine','PPODE','Parareal'},'Interpreter','latex','numcolumns',3,'location','northwest')
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

prob_error2 = abs(F_sol(:,2) - u_prob(:,2)); prob_error2(prob_error2==0) = 10^(-40);
para_error2 = abs(F_sol(:,2) - u_para(:,2)); para_error2(para_error2==0) = 10^(-40);

prob_error3 = abs(F_sol(:,3) - u_prob(:,3)); prob_error3(prob_error3==0) = 10^(-40);
para_error3 = abs(F_sol(:,3) - u_para(:,3)); para_error3(para_error3==0) = 10^(-40);

figure(3)
subplot(3,1,1)
hold on
plot(t_fine,para_error1,'r','LineWidth',1.2)
plot(t_fine,prob_error1,'b','LineWidth',1.2)
hold off
set(gca,'yscale','log')
ylim([10^(-15) 10^(0)]);
xlabel('$t$','interpreter','latex'); ylabel('Error','interpreter','latex');
box on; grid on;

subplot(3,1,2)
hold on
plot(t_fine,para_error2,'r','LineWidth',1.2)
plot(t_fine,prob_error2,'b','LineWidth',1.2)
hold off
%set(gca,'xtick',[])
set(gca,'yscale','log')
ylim([10^(-15) 10^(0)]);
xlabel('$t$','interpreter','latex'); ylabel('Error','interpreter','latex');
box on; grid on;

subplot(3,1,3)
hold on
plot(t_fine,para_error3,'r','LineWidth',1.2)
plot(t_fine,prob_error3,'b','LineWidth',1.2)
hold off
set(gca,'yscale','log')
ylim([10^(-15) 10^(0)]);
xlabel('$t$','interpreter','latex'); ylabel('Error','interpreter','latex');
box on; grid on;
legend({'Fine - Parareal','Fine - PPODE'},'interpreter','latex','location','southeast')




