function     [ lamb, TX , C, A, U ]=shapelambda2D_4( coef, inx1, inx2, N, lamb5, TX5, C5, A5,...
    lamb5_4, TX5_4, C5_4, A5_4, lamb6, TX6, C6, A6,...
    lamb6_5, TX6_5, C6_5, A6_5, lamb7, TX7, C7, A7,...
    lamb7_6, TX7_6, C7_6, A7_6, lamb8, TX8, C8, A8,...
    lamb8_7, TX8_7, C8_7, A8_7 )

cha=inx2-inx1;
Num=prod(N);
t=linspace(inx1(1,1),inx2(1,1),N(1,1));
x=linspace(inx1(1,2),inx2(1,2),N(1,2));

C=coef*cha;
A=N-1;

[X,Y]=meshgrid(t,x);
TX=[X(:) Y(:)];

U=iniPDE(TX);

[ FAI, FAI_t, FAI_xx ] = mq2d( TX, TX, A, C );

P=FAI_t - FAI_xx;

U = U - PDE( TX,lamb5, TX5, C5, A5, lamb5_4, TX5_4, C5_4, A5_4 )-...
        PDE( TX, lamb6, TX6, C6, A6, lamb6_5, TX6_5, C6_5, A6_5 )-...
        PDE( TX, lamb7, TX7, C7, A7, lamb7_6, TX7_6, C7_6, A7_6 )-...
        PDE( TX, lamb8, TX8, C8, A8, lamb8_7, TX8_7, C8_7, A8_7 );

for i=1:Num

    if (TX(i,1)-inx1(1,1))*(TX(i,2)-inx1(1,2))*(TX(i,2)-inx2(1,2)) == 0

        P(i,:)=  FAI(i,:);
        
        U(i)=boundcondition(TX(i,:))-...
            (inner_test(TX(i,:),lamb5,TX5,C5,A5)-inner_test(TX(i,:),lamb5_4,TX5_4,C5_4,A5_4))-...
            (inner_test(TX(i,:),lamb6,TX6,C6,A6)-inner_test(TX(i,:),lamb6_5,TX6_5,C6_5,A6_5))-...
            (inner_test(TX(i,:),lamb7,TX7,C7,A7)-inner_test(TX(i,:),lamb7_6,TX7_6,C7_6,A7_6))-...
            (inner_test(TX(i,:),lamb8,TX8,C8,A8)-inner_test(TX(i,:),lamb8_7,TX8_7,C8_7,A8_7));  % boundary condition
    end

end

% cond(P)
[F,J]=lu(P);
Jlamda=F\U;
lamb=J\Jlamda;

end
