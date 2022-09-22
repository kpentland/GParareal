function kernel = GPoptimise(x,y,kernel)
% Carries out optimisation of the hyperparameters specified in 'kernel' 
% structure. 

%INPUTS:
% x:         training points (N x d)
% y:         observations at training points (N x 1)
% kernel:    structure containing covariance function and hyperparameters

%OUTPUTS:
% kernel:    structure containing covariance function and optimised hyperparameters


theta = kernel.theta;    %initial hyperparameters

%maximise log marginal likelihood using 'fminsearch' function
theta1 = fminsearch(@(theta)log_likelihood(x,y,kernel,theta),theta,optimset('TolFun',1e-6,'TolX',1e-6,'Display','off'));
% theta1 = fmincon(@(theta)log_likelihood(x,y,kernel,theta),theta,[],[],[],[],zeros(length(theta),1),1e10*ones(length(theta),1),[]); %,optimoptions('fmincon','SpecifyObjectiveGradient',true));

kernel.theta = theta1;  %optimised hyperparameters

end




function log_lik = log_likelihood(x,y,kernel,theta)
% Evaluates the (negative) log likelihood function for the current kernel
% hyperparams (as well as its gradient if available).

%calculate covariance matrix
K = kernel.func(x,x,theta(1),theta(2));
N = size(x,1);

%matrix inverse computations (from Rasmussen/Williams 2004)
L = chol(K + eye(N)*kernel.jitter,'lower');
alph = L'\(L\y);

%(negative) log marginal likelihood
log_lik = -( -0.5*y'*alph - sum( log ( diag(L) ) ) - (N/2)*log(2*pi) );

% if nargout > 1 % gradient required
%     b = alph*alph';
%     Kinv = inv(K);
%     grad_log_lik = -0.5*[trace((b-Kinv)*isoSE_cov_dsigmax(x,x,theta(1),theta(2))); 
%         trace((b-Kinv)*isoSE_cov_dsigmay(x,x,theta(1),theta(2)))];
% end

end







% function Kx = isoSE_cov_dsigmax(xi,xj,sigma_x,sigma_y)
% %partial deriv of covariance kernel.
% 
% d = size(xi,2);
% 
% if d == 1
%     Kx = ((sigma_y^2)/(sigma_x^3)).*((xi-xj').^2).*exp(-0.5*(1/(sigma_x^2))*((xi-xj').^2));
% else
%     xd = pdist2(xi,xj,'squaredeuclidean');
%     Kx = ((sigma_y^2)/(sigma_x^3)).*xd.*exp(-0.5*(1/(sigma_x^2)).*xd);
% end
% 
% end
% 
% function Ky = isoSE_cov_dsigmay(xi,xj,sigma_x,sigma_y)
% %partial deriv of covariance kernel.
% 
% d = size(xi,2);
% 
% if d == 1
%     Ky = 2*sigma_y*exp(-0.5*(1/(sigma_x^2))*((xi-xj').^2));
% else
%     Ky = 2*sigma_y*exp(-0.5*(1/(sigma_x^2))*pdist2(xi,xj,'squaredeuclidean'));
% end
% 
% end
