function [ U ] = PPP( X )
% approximated initial condition at an earlier time \tau
A = load('Smoothinitial');
x1 = A.X_ini;
U1 = A.U_ini;
I = X(1,2)==x1;
U = U1(I);

end

