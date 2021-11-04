%% Non-autonomous Equation (1D)
%Here we solve a non-autonomous nonlinear ODE. Plots are described below.  
%To test, run the following script in its entirety. 


clear; close all; clc;

% INPUTS
f = @(t,u)(u*cos(u*t)+cos(t));     %the ODE to be solved for parareal
tspan = [0,15];                    %time interval of integration
u0 = 1;                            %initial condition for ODE system + initial time of integration
auto = 0;                          %is the system autonomous (yes=1, no=0)
epsilon = 10^(-8);                 %error tolerance
N = 30;                            %no. of time sub-intervals (processors)
Ng = 300;                           %coarse time steps
Nf = Ng*500;                       %fine time steps
G = 'RK1';                         %coarse solver
F = 'RK4';                         %fine solver

% Gaussian Process prior inputs
K = gpcf_sexp('lengthScale',[1,1], 'magnSigma2',1);  %square exponential kernel
like = lik_gaussian();                           %Gaussian likelihood with prior scale parameter

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
xlabel('$t$','interpreter','latex'); ylabel('$u$ error','interpreter','latex');
box on; grid on;
legend({'Fine - Parareal','Fine - PPODE'},'interpreter','latex','location','northwest')



%Fig 4: mean of the GP regression over (F-G)(x)

% Choose test points on a dense grid
ns = 20^2;
xs1 = linspace(tspan(1),tspan(2),sqrt(ns));
xs2 = linspace(min(u_prob(:,1)),max(u_prob(:,1)),sqrt(ns));
[xs1,xs2] = meshgrid(xs1,xs2); 

[means, vars] = gp_pred(gp{1},xx,DD,[xs1(:),xs2(:)]);

% Plot the prediction and data
figure(4)
clf
mesh(xs1, xs2, reshape(means,sqrt(ns),sqrt(ns)));
hold on
plot3(xx(:,1),xx(:,2),DD(:),'.r')
axis on;
xlim([tspan(1),tspan(2)])
ylim([min(u_prob(:,1)),max(u_prob(:,1))])
xlabel('Time, $t$','interpreter','latex'); ylabel('Initial value, $u$','interpreter','latex');
zlabel('$(\mathcal{F}-\mathcal{G}) (t,u)$','interpreter','latex');
