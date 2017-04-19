function [ U ] = iniPDE( X )
% initial PDE operator value for particular PDE
N=length(X);

% f = sin(pi*t) * sin(pi*x) * sin(pi*y) * sin(pi*z); target function
t=X(:,1);
x=X(:,2);
y=X(:,3);
z=X(:,4);

U = pi*cos(pi*t).*sin(pi*x).*sin(pi*y).*sin(pi*z) +...
    3*pi^2*sin(pi*t).*sin(pi*x).*sin(pi*y).*sin(pi*z);

end

