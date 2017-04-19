function     [ lamb, TX , C, A, U ]=shapelambda2D_3(coef, tsec,r,sigma,T,E,inx1,inx2,N,lamb2, TX2, C2, A2,lamb3, TX3, C3, A3,...
    lamb_3, TX_3, C_3, A_3,lamb4, TX4, C4, A4,lamb_4, TX_4, C_4, A_4,lamb5, TX5, C5, A5)

Num=prod(N);
t=linspace(0,tsec,N(1,1));
x=linspace(inx1,inx2,N(1,2));
h1=coef*tsec;
h2=coef*(inx2-inx1);
C=[h1, h2];
A=N-1;

[XXX, YYY] = meshgrid(t, x);
TX = [XXX(:) YYY(:)];

U=zeros(Num,1);

[ FAI, FAI_t, FAI_x, FAI_xx ] = mq2d( TX, TX, A, C );

P = FAI_t + sigma^2*FAI_xx/2 + r*FAI_x - r*FAI;

U= U - PDE( TX,r,sigma,lamb2,TX2,C2,A2,lamb3,TX3,C3,A3  ) -...
    PDE( TX,r,sigma,lamb_3,TX_3,C_3,A_3,lamb4,TX4,C4,A4  ) -...
    PDE( TX,r,sigma,lamb_4,TX_4,C_4,A_4,lamb5,TX5,C5,A5  );

for i=1:Num


    if TX(i,2) == inx1 || TX(i,2) == inx2
        
        P(i,:) = FAI(i,:);        

        U(i)=max( 0, TX(i,2) - E*exp( -r * ( T - TX(i,1) ) ) )-...
            (inner_test(TX(i,1),TX(i,2),lamb3,TX3,C3,A3)-inner_test(TX(i,1),TX(i,2),lamb2,TX2,C2,A2))-...
            (inner_test(TX(i,1),TX(i,2),lamb4,TX4,C4,A4)-inner_test(TX(i,1),TX(i,2),lamb_3,TX_3,C_3,A_3))-...
            (inner_test(TX(i,1),TX(i,2),lamb5,TX5,C5,A5)-inner_test(TX(i,1),TX(i,2),lamb_4,TX_4,C_4,A_4));  % boundary condition

    end
        
    if TX(i,1) == tsec
       
        P(i,:) = FAI(i,:);
        U(i) = PPP( TX(i,:) ) - ...
            (inner_test(TX(i,1),TX(i,2),lamb3,TX3,C3,A3)-inner_test(TX(i,1),TX(i,2),lamb2,TX2,C2,A2))-...
            (inner_test(TX(i,1),TX(i,2),lamb4,TX4,C4,A4)-inner_test(TX(i,1),TX(i,2),lamb_3,TX_3,C_3,A_3))-...
            (inner_test(TX(i,1),TX(i,2),lamb5,TX5,C5,A5)-inner_test(TX(i,1),TX(i,2),lamb_4,TX_4,C_4,A_4));
    end

end

% cond(P)
[F,J]=lu(P);
Jlamda=F\U;
lamb=J\Jlamda;


end




