%% Brusselator system (2D)
%Here we solve the Brusselator system - a pair of coupled stiff ODEs. Plots are described below.  
%To test, run the following script in its entirety. 

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
xlabel('$u_1$','interpreter','latex'); ylabel('$u_2$','interpreter','latex');
zlabel('$(\mathcal{F}-\mathcal{G})_1 (u_1,u_2)$','interpreter','latex');
title('Output one');

figure(5)
clf
mesh(xs1, xs2, reshape(indep_means{2},sqrt(ns),sqrt(ns)));
hold on
plot3(xx(:,1),xx(:,2),DD(:,2),'.r')
axis([min(xx(:,1)) max(xx(:,1)) min(xx(:,2)) max(xx(:,2))])
axis on;
xlabel('$u_1$','interpreter','latex'); ylabel('$u_2$','interpreter','latex');
zlabel('$(\mathcal{F}-\mathcal{G})_2 (u_1,u_2)$','interpreter','latex');
title('Output two');
