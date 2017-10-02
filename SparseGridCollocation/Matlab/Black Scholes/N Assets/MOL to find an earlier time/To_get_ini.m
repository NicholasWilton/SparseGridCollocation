% To get estimations at useful nodes in [Smin, Sman] at time (T-Tdone)

Eurocalloption1D

Smin = 0;
Smax = 3*E;
X_ini = linspace(Smin,Smax,2^15+1)';
WriteAllToFile("X_ini.txt", X_ini);
phi = ones(length(X_ini),length(x));
for i = 1 : length(x)
    [phi(:,i)] = mq2d(X_ini,x(i),c(i));
end
WriteAllToFile("lamb.final.txt", lamb);
WriteAllToFile("phi.txt", phi);
U_ini = phi * lamb;
WriteAllToFile("U_ini.txt", U_ini);
save Smoothinitial.mat X_ini U_ini Tdone