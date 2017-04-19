function [ U ] = iniPDE( X )
% initial PDE operator value for particular PDE

N=length(X);

% # f = exp(-pi^2*t)*sin(pi*x) # target function
% # ft - fxx #  PDE operator
% x=X(:,1);
% y=X(:,2);
U = X(:,1).*0;

end

