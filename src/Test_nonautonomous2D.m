%% Non-autonomous system (2D)
%Here we solve a non-autonomous ODE system of two equations. Plots are described below.  
%To test, run the following script in its entirety.

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
K = cell(2,1);                                          %covariance kernel structure required for each output
K{1} = gpcf_sexp('lengthScale',1, 'magnSigma2',1);      %square exponential kernel (isotropic)
K{2} = gpcf_sexp('lengthScale',1, 'magnSigma2',1);      %square exponential kernel (isotropic)
like = lik_gaussian();                                  %Gaussian likelihood with prior scale parameter


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
xlabel('$t$','Interpreter','latex'); ylabel('$u_1$','Interpreter','latex'); zlabel('$u_2$','Interpreter','latex');
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


