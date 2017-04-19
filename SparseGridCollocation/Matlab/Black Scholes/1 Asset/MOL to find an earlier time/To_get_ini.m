% To get estimations at useful nodes in [Smin, Sman] at time (T-Tdone)

Eurocalloption1D

Smin = 0;
Smax = 3*E;
X_ini = linspace(Smin,Smax,2^15+1)';
phi = ones(length(X_ini),length(x));
for i = 1 : length(x)
    [phi(:,i)] = mq2d(X_ini,x(i),c(i));
end
U_ini = phi * lamb;

save Smoothinitial.mat X_ini U_ini Tdone