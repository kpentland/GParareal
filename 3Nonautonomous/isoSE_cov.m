function K = isoSE_cov(xi,xj,sigma_x,sigma_y)
%Builds the (isotropic) Matern covariance matrix (need to select which
% kernel to use by uncommenting the required "K = ..." line below).

%INPUTS:
% xi:         input points (N x d)
% xj:         input points (M x d)
% sigma_x:    input length scale (postive real number)
% sigma_y:    output length scale (positive real number)

%OUTPUTS:
% K:          covariance matrix (N x M)

% Matern 1/2 kernel
% r = pdist2(xi,xj,'euclidean');
% K = (sigma_y^2)*exp(-(1/sigma_x)*r);

% Matern 3/2 kernel
% r = pdist2(xi,xj,'euclidean');
% K = (sigma_y^2)*(1 + (sqrt(3)/sigma_x)*r)*exp(-sqrt(3)*r/sigma_x);

% Matern inf kernel (i.e. SE kernel)
r = pdist2(xi,xj,'squaredeuclidean');
K = (sigma_y^2)*exp(-0.5*(1/(sigma_x^2))*r);

end