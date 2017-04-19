% This matlab code is to find a time (Tdone) and determine an 
% estimation at an earlier time (T-Tdone) by radial basis function
% approximation on one asset European option problem
% Later we will apply sparse grid on the domain [0,T-Tdone] * [Smin, Smax]
% Implementation : Method of lines
%                  Time-stepping: \theta method


T = 1; % T means maturity time
Tdone = 0; 
Tend = 0.2*T; % Tend means maximum value for Tdone
dt = 1/1000; % time step
E = 100; % strike price
r = 0.03; % rate
sig = 0.15; % sigma
theta = 0.5; % choose \theta = 0.5
inx1 = -E; % stock price domain is determined as [inx1, inx2]
inx2 = 6*E;

initcond = @(x) max(x-E,0); % payoff function at T
N_uniform = 5000; % a large number of uniform nodes
x = linspace(inx1,inx2,N_uniform)';
u0 = initcond(x);
dx = diff(x); 
c = 2*min([Inf;dx],[dx;Inf]);  % shape parameter c = 2h, 
                               % h is the distance between two adjacent nodes

xx = linspace(0,3*E,1000)'; % testing points on [Smin = 0, Sman = 3E].

IT = x >= 0.8*E & x <= 1.2*E; % find the nodes around strike price
ArroundE = x(IT);

N=length(x);
[D1_mid, D2_mid, D3_mid] = deal(zeros(length(ArroundE),N));
[A, D1, D2] = deal(zeros(N,N));
Axx = zeros(length(xx),N);
for j=1:N
    [Axx(:,j)] = mq2d(xx,x(j),c(j));
    [A(:,j),D1(:,j),D2(:,j)] = mq2d(x,x(j),c(j));
    [~,D1_mid(:,j),D2_mid(:,j),D3_mid(:,j)] = mq2d(ArroundE,x(j),c(j));
    D1_mid(:,j) = D1_mid(:,j) ./ ArroundE;
    D2_mid(:,j) = D2_mid(:,j) ./ (ArroundE).^2;
end
lamb = A\u0;
uu0 = Axx*lamb;
deri1 = D1_mid*lamb;
deri2 = D2_mid*lamb;
deri3 = D3_mid*lamb;
A1 = A(1,:);
Aend = A(end,:);
A(1,:) = [];
A(end,:) = [];
D1(1,:) = [];
D1(end,:) = [];
D2(1,:) = [];
D2(end,:) = [];

[ Price ] = ECP( [ones(length(xx),1).*(T-Tdone),xx], r, sig, T, E);
% % % % % % % % % % 
han = plot(xx,uu0-Price,'-ko',xx,0*xx.^0,'ko','MarkerFaceColor','k','MarkerSize',2);
tp = title('','erasemode','xor'); hold on;
xlabel('Stock price');ylabel('V^{*}_{app} - C');
set(tp,'string',sprintf('T = %.3f,    N = %3i.',Tdone,N))
% % % % % % % % % % % % 
P = A*r - 0.5*(sig^2)*(D2) - r*(D1); 
H=A+dt*(1-theta)*P;  % D
G=A-dt*theta*P;      % B
while Tend - Tdone > 10^(-8)    
Tdone = Tdone + dt;
fff = [0;G*lamb;inx2 - exp(-r*Tdone)*E];
HH = [A1;H;Aend];
lamb=HH\fff;

uu0 = Axx*lamb;
deri1 = D1_mid*lamb;
deri2 = D2_mid*lamb;
deri3 = D3_mid*lamb;

d1=(log(ArroundE./E)+(r+sig^2/2)*(Tdone))/sig/sqrt(Tdone);
d2=d1-sig*sqrt(Tdone);
E_Delta=normcdf(d1);
E_Gamma=exp(-d1.^2/2)/sqrt(2*pi)./sig/sqrt(Tdone)./ArroundE;
E_Speed=-E_Gamma./ArroundE.*(d1/sig/sqrt(Tdone)+1);
[ Price ] = ECP( [ones(length(xx),1).*(T-Tdone),xx], r, sig, T, E);
% % % % % % % % % 
xdata = num2cell([xx, xx]',2);
ydata = num2cell([uu0-Price, 0*xx.^0]',2);
set(han,{'xdata'},xdata,{'ydata'},ydata);
drawnow
set(tp,'string',sprintf('T = %.3f,    N = %3i.',Tdone,N))
% % % % % % % % % 
% to find a time Tdone that satisfies line 123 - 127

Ptop = max(deri3); % peak top of Speed approximation
Pend = min(deri3); % peak end of Speed approximation
I1 = find(deri3 == Ptop);
I2 = find(deri3 == Pend);
a = min(I1,I2);
b = max(I1,I2);
part1 = diff(deri3(1:a));
part2 = diff(deri3(a:b));
part3 = diff(deri3(b:end));
II1 = part1 >= 0;
II2 = part2 >= 0;
II3 = part3 >= 0;
if min(deri2) >= 0 % Gamma greater than 0
%  Approximation of Speed is monotonic in subintervals
if sum(II1) == 0 || sum(II1) == length(part1)
    if sum(II2) == 0 || sum(II2) == length(part2)
        if sum(II3) == 0 || sum(II3) == length(part3)
            disp(Tdone)
            break
        end
    end
end
end
% % % % % % % % % % % 

end

save Approximation.mat Tdone x u0 c A xx T E r sig lamb



