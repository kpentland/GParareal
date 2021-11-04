%% Logistic Equation (1D)
%Here we solve the Logistic equation with growth rate r = 1. Plots are described below.  
%To test, run the following script in its entirety. 

clear; close all; clc;

% INPUTS
u0 = 5;                            %initial condition for ODE
f = @(t,u)( 1*u*(1-u) );           %the ODE to be solved
auto = 1;                          %is the system autonomous (yes=1, no=0)
tspan = [0,15];                    %time interval of integration
epsilon = 10^(-8);                 %error tolerance
N = 20;                            %no. of time sub-intervals (processors)
Ng = 80;                           %coarse time steps
Nf = 1000*Ng;                      %fine time steps
G = 'RK1';                         %coarse solver
F = 'RK4';                         %fine solver

% GP prior inputs

% Gaussian Process prior inputs
K = cell(1,1);                                          %covariance kernel structure required for each output
K{1} = gpcf_sexp('lengthScale',1,'magnSigma2',1);       %square exponential covariance kernel
like = lik_gaussian();                                  %Gaussian likelihood with prior scale parameter


%%PLOTS

%fine time step and mesh
dt = (tspan(2)-tspan(1))/Nf;
t_fine = (tspan(1):dt:tspan(end));


% Fig 1: PPODE, fine, parareal solutions vs. analytical
[t,u,err,k,gp,xx,DD] = PPODE(f,auto,tspan,u0,N,Ng,Nf,epsilon,F,G,K,like);     % PPODE solution
[~,F_sol] = RK(t_fine,u0,f,F);                                                % fine solution
[~,u_p,err_para,k_para] = parareal(f,tspan,u0,N,Ng,Nf,epsilon,F,G);           % parareal solution
u_true = (u0*exp(1*t_fine)./(1 - u0 + u0*exp(1*t_fine)))';                    % analytical solution

% display optimised GP hyperparameters
fprintf('\n Optimised GP hyperparameters: \n')
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
%yticks((-0.6:0.2:0.6))
%ylim([-0.6 0.6])
legend({'$\pm$ two standard deviations','Mean','Observations'},'interpreter','latex','location','southeast')
xlabel('Initial value, $u$','interpreter','latex'); ylabel('$(\mathcal{F}-\mathcal{G})(u)$','interpreter','latex');
grid on; box on;

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

