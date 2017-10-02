 N=length(x);
 IT = x >= 0.8*E & x <= 1.2*E;
ArroundE = x(IT);
[D1_mid, D2_mid, D3_mid] = deal(zeros(length(ArroundE),N));
[A, D1, D2] = deal(zeros(N,N));
for j=1:N
    [Axx(:,j)] = mq2d(xx,x(j),c(j));
    [A(:,j),D1(:,j),D2(:,j)] = mq2d(x,x(j),c(j));
    [~,D1_mid(:,j),D2_mid(:,j),D3_mid(:,j)] = mq2d(ArroundE,x(j),c(j));
    D1_mid(:,j) = D1_mid(:,j) ./ ArroundE;
    D2_mid(:,j) = D2_mid(:,j) ./ (ArroundE).^2;
end

uu0 = Axx*lamb;
deri2 = D2_mid*lamb;
deri3 = D3_mid*lamb;
d1=(log(ArroundE./E)+(r+sig^2/2)*(Tdone))/sig/sqrt(Tdone);
d2=d1-sig*sqrt(Tdone);
E_Delta=normcdf(d1);
E_Gamma=exp(-d1.^2/2)/sqrt(2*pi)./sig/sqrt(Tdone)./ArroundE;
[ Pal ] = ECP( [ones(length(xx),1).*(T-Tdone),xx], r, sig, T, E);
E_Speed=-E_Gamma./ArroundE.*(d1/sig/sqrt(Tdone)+1);

figure;plot(ArroundE,deri2-E_Gamma,'-ko',ArroundE,0*ArroundE.^0,'ko','MarkerFaceColor','k','MarkerSize',2);
tp = title('','erasemode','xor'); hold on;
xlabel('S');ylabel('Gamma error');
set(tp,'string',sprintf('t = %.3f,    N = %3i.',T-Tdone,length(ArroundE)))

figure;plot(ArroundE,deri3-E_Speed,'-ko',ArroundE,0*ArroundE.^0,'ko','MarkerFaceColor','k','MarkerSize',2);
tp = title('','erasemode','xor'); hold on;
xlabel('S');ylabel('Speed error');
set(tp,'string',sprintf('t = %.3f,    N = %3i.',T-Tdone,length(ArroundE)))

figure;plot(xx,uu0-Pal,'-ko',xx,0*xx.^0,'ko','MarkerFaceColor','k','MarkerSize',2);
tp = title('','erasemode','xor'); hold on;
xlabel('S');ylabel('Value error');
set(tp,'string',sprintf('t = %.3f,    N = %3i.',T-Tdone,length(xx)))

figure;plot(ArroundE,deri2,'-ko',ArroundE,0*ArroundE.^0,'ko','MarkerFaceColor','k','MarkerSize',2);
tp = title('','erasemode','xor'); hold on;
xlabel('S');ylabel('Gamma');
set(tp,'string',sprintf('t = %.3f,    N = %3i.',T-Tdone,length(ArroundE)))

figure;plot(ArroundE,deri3,'-ko',ArroundE,0*ArroundE.^0,'ko','MarkerFaceColor','k','MarkerSize',2);
tp = title('','erasemode','xor'); hold on;
xlabel('S');ylabel('Speed');
set(tp,'string',sprintf('t = %.3f,    N = %3i.',T-Tdone,length(ArroundE)))

figure;plot(xx,uu0,'-ko',xx,0*xx.^0,'ko','MarkerFaceColor','k','MarkerSize',2);
tp = title('','erasemode','xor'); hold on;
xlabel('S');ylabel('Option Value');
set(tp,'string',sprintf('t = %.3f,    N = %3i.',T-Tdone,length(xx)))