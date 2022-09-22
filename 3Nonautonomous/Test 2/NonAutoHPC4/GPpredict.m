function [post_mean, post_cov] = GPpredict(x,y,x_star,kernel)
% Carries out prediction for the GP emulator on test points x_star given
% training data (x,y) and covariance structure 'kernel'.

%INPUTS:
% x:         training points (N x d)
% y:         observations at training points (N x 1)
% x_star:    test points (M x d)
% kernel:    structure containing covariance function and hyperparameters

%OUTPUTS:
% post_mean: posterior mean predicition at test points
% post_cov:  posterior covariance prediction between test points


%calculate covariance matrix blocks
K = kernel.func(x,x,kernel.theta(1),kernel.theta(2));
K_star = kernel.func(x,x_star,kernel.theta(1),kernel.theta(2));
K_star_star = kernel.func(x_star,x_star,kernel.theta(1),kernel.theta(2));
N = size(x,1);

%matrix inverse computations (from Rasmussen/Williams 2004)
L = chol(K + eye(N)*kernel.jitter,'lower');
alph = L'\(L\y);
v = L\K_star;

%posterior mean and covariance
post_mean = K_star'*alph;
post_cov = K_star_star - v'*v;  post_cov = (post_cov + post_cov')/2;

end