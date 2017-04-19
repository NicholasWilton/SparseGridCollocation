function [ U ] = iniPDE( X )
% initial PDE operator value for particular PDE
N=length(X);

% # f = exp(-2*pi^2 .* t) .* sin(pi .* x) .* cos(pi .* y) # target function
% # ft - fxx - fyy #  PDE operator
t=X(:,1);
x=X(:,2);
y=X(:,3);
% U = X(:,1).*0;

% f = sin(pi*x1)*sin(pi*x2)*sin(pi*t)
% # ft - fxx - fyy #  PDE operator
U = pi.*cos(pi.*t).*sin(pi.*x).*sin(pi.*y) +...
    2*pi^2.*sin(pi.*t).*sin(pi.*x).*sin(pi.*y);


end

