% This matlab code is to find a time (Tdone) and determine an 
% estimation at an earlier time (T-Tdone) by radial basis function
% approximation on one asset European option problem
% Later we will apply sparse grid on the domain [0,T-Tdone] * [Smin, Smax]
% Implementation : Method of lines
%                  Time-stepping: \theta method


T = 1; % T means maturity time
Tdone = 0; 
Tend = 0.5*T; % Tend means maximum value for Tdone
dt = 1/1000; % time step
E = 100; % strike price
r = 0.03; % rate
sig = 0.15; % sigma
theta = 0.5; % choose \theta = 0.5
assets = 2;
[inx1, inx2, testNodes, centralNodes, aroundStrikeNodes] = SetupBasket(assets, 50, 50, 0.3, E);

%initcond = @(x) max(x-E,0); % payoff function at T

u0 = PayoffFunction(testNodes, E);

dx = DiffN(testNodes); 
c = 2*min([Inf;dx],[dx;Inf]);  % shape parameter c = 2h, 
                               % h is the distance between two adjacent nodes

[N,colN]=size(testNodes);
[n,~]=size(aroundStrikeNodes);
a = ones(N,colN);

[D_mid, D1_mid, D2_mid] = deal(zeros(n,N));
[A, D1, D2] = deal(zeros(N,N));
Axx = zeros(length(centralNodes),N);
S = PayoffS(aroundStrikeNodes);
Sc = PayoffS(centralNodes);
for j=1:N
    [Axx(:,j)] = mqnd(centralNodes,testNodes(j,:),a(j,:),c(j,:), assets);
    [A(:,j),d1,d2_upperT,d2_diag] = mqnd(testNodes,testNodes(j,:),a(j,:), c(j,:), assets);
    for m=d1
      D1(:,j) = D1(:,j) + m{1};  end
    for m=d2_diag
      D2(:,j) = D2(:,j) + m{1};  end
    for m=d2_upperT
      D2(:,j) = D2(:,j) +  2 * m{1};  end

    [D_mid,d1_mid,d2_upperT_mid,d2_diag_mid] = mqnd(aroundStrikeNodes,testNodes(j,:),a(j,:), c(j,:), assets);
    for m=d1_mid
      D1_mid(:,j) = D1_mid(:,j) + m{1};  end
    for m=d2_diag_mid
      D2_mid(:,j) = D2_mid(:,j) + m{1};  end
    for m=d2_upperT_mid
      D2_mid(:,j) = D2_mid(:,j) + 2* m{1};  end
    
    D1_mid(:,j) = D1_mid(:,j) ./ S;
    D2_mid(:,j) = D2_mid(:,j) ./ S.^2;
end
[speedRows,speedCols]=size(D2_mid);

%D_Speed = deal(ones(speedRows,speedCols));
D3_mid = - D2_mid ./ S;
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

[ Price ] = ECP( [ones(length(Sc),1).*(T-Tdone),Sc], r, sig, T, E);
% % % % % % % % % % 
% figure
% han = plot(Sc,uu0-Price,'-ko',Sc,0*Sc.^0,'ko','MarkerFaceColor','k','MarkerSize',2);
% tp = title('','erasemode','xor'); hold on;
% xlabel('Stock price');ylabel('V^{*}_{app} - C');
% set(tp,'string',sprintf('Error T = %.3f,    N = %3i.',Tdone,N))
% % % % % % % % % % % % 
figure
han1 = plot(S,deri2,'-ko',S,0*S.^0,'ko','MarkerFaceColor','k','MarkerSize',2);
tp1 = title('','erasemode','xor'); hold on;
xlabel('Stock price');ylabel('Gamma');
set(tp1,'string',sprintf('Gamma T = %.3f,    N = %3i.',Tdone,N))
% % % % % % % % % % % % 
figure
han2 = plot(S,deri3,'-ko',S,0*S.^0,'ko','MarkerFaceColor','k','MarkerSize',2);
tp2 = title('','erasemode','xor'); hold on;
xlabel('Stock price');ylabel('Speed');
set(tp2,'string',sprintf('Speed T = %.3f,    N = %3i.',Tdone,N))
% % % % % % % % % % % % 
P = A*r - 0.5*(sig^2)*(D2) - r*(D1); 
H=A+dt*(1-theta)*P;  % D
G=A-dt*theta*P;      % B
count =0;
while Tend - Tdone > 10^(-8)    
Tdone = Tdone + dt;
first= 0;
last = mean(inx2(1,:));
fff = [first;G*lamb;last - exp(-r*Tdone)*E];
HH = [A1;H;Aend];
lamb=HH\fff;

uu0 = Axx*lamb;
deri1 = D1_mid*lamb;
deri2 = D2_mid*lamb;
deri3 = D3_mid*lamb;

d1=(log(S./E)+(r+sig^2/2)*(Tdone))/sig/sqrt(Tdone);
d2=d1-sig*sqrt(Tdone);
E_Delta=normcdf(d1);
E_Gamma=exp(-d1.^2/2)/sqrt(2*pi)./sig/sqrt(Tdone)./S;
E_Speed=-E_Gamma./S.*(d1/sig/sqrt(Tdone)+1);
deri3 = deri3 .*  (d1/sig/sqrt(Tdone)+1);
[ Price ] = ECP( [ones(length(centralNodes),1).*(T-Tdone),centralNodes], r, sig, T, E);
% % % % % % % % % 
% xdata = num2cell([S, S]',2);
% ydata = num2cell([uu0-Price, 0*Sc.^0]',2);
% set(han,{'xdata'},xdata,{'ydata'},ydata);
% drawnow
% set(tp,'string',sprintf('Error T = %.3f,    N = %3i. min = %.8f',Tdone,N, min(deri2)))
% % % % % % % % % 
plot1 = [deri2,S];
plot1 = sortrows(plot1, 2);
plot1 = TakeMeans(plot1);
xdata1 = num2cell([plot1(:,2), plot1(:,2)]',2);
ydata1 = num2cell([plot1(:,1), 0*plot1(:,2).^0]',2);
set(han1,{'xdata'},xdata1,{'ydata'},ydata1);
drawnow
set(tp1,'string',sprintf('Gamma T = %.3f,    N = %3i. min = %.8f',Tdone,N, min(deri2)))
% % % % % % % % % 
plot2 = [deri3,S];
plot2 = sortrows(plot2, 2);
plot2 = TakeMeans(plot2);
xdata2 = num2cell([plot2(:,2), plot2(:,2)]',2);
ydata2 = num2cell([plot2(:,1), 0*plot2(:,2).^0]',2);
set(han2,{'xdata'},xdata2,{'ydata'},ydata2);
drawnow
set(tp2,'string',sprintf('Speed T = %.3f,    N = %3i. min = %.8f',Tdone,N, min(deri2)))
% % % % % % % % % 
% to find a time Tdone that satisfies line 123 - 127
deri2 = plot1(:,1);
deri3 = plot2(:,1);
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
%set(tp,'string',sprintf('T = %.3f,    N = %3i. min = %.8f sum(II2) = %i',Tdone,N, min(deri2), sum(II2)))
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
count = count + 1;
end

save Approximation.mat Tdone testNodes u0 c A centralNodes T E r sig lamb



