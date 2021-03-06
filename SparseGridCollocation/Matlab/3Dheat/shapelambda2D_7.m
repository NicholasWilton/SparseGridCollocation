function     [ lamb, TXY , C, A, U ]= shapelambda2D_7( coef, inx1, inx2, N, lamb5, TX5, C5, A5,...
    lamb5_4, TX5_4, C5_4, A5_4, lamb5_3, TX5_3, C5_3, A5_3,lamb6, TX6, C6, A6,...
    lamb6_5, TX6_5, C6_5, A6_5, lamb6_4, TX6_4, C6_4, A6_4, lamb7, TX7, C7, A7,...
    lamb7_6, TX7_6, C7_6, A7_6, lamb7_5, TX7_5, C7_5, A7_5, lamb8, TX8, C8, A8,...
    lamb8_7, TX8_7, C8_7, A8_7, lamb8_6, TX8_6, C8_6, A8_6, lamb9, TX9, C9, A9,...
    lamb9_8, TX9_8, C9_8, A9_8, lamb9_7, TX9_7, C9_7, A9_7, lamb10, TX10, C10, A10,...
    lamb10_9, TX10_9, C10_9, A10_9, lamb10_8, TX10_8, C10_8, A10_8, lamb11, TX11, C11, A11,...
    lamb11_10, TX11_10, C11_10, A11_10, lamb11_9, TX11_9, C11_9, A11_9 )

cha=inx2-inx1;
Num=prod(N);
t=linspace(inx1(1,1),inx2(1,1),N(1,1));
x=linspace(inx1(1,2),inx2(1,2),N(1,2));
y=linspace(inx1(1,3),inx2(1,3),N(1,3));

C=coef*cha;
A=N-1;

[X,Y,Z]=meshgrid(t,x,y);
TXY=[X(:) Y(:) Z(:)];

U=iniPDE(TXY);

[ FAI, FAI_t, FAI_xx, FAI_yy ] = mq3d( TXY, TXY, A, C );

P=FAI_t - FAI_xx - FAI_yy;

U = U - PDE( TXY, lamb5, TX5, C5, A5, lamb5_4, TX5_4, C5_4, A5_4, lamb5_3, TX5_3, C5_3, A5_3 )-...
        PDE( TXY, lamb6, TX6, C6, A6, lamb6_5, TX6_5, C6_5, A6_5, lamb6_4, TX6_4, C6_4, A6_4 )-...
        PDE( TXY, lamb7, TX7, C7, A7, lamb7_6, TX7_6, C7_6, A7_6, lamb7_5, TX7_5, C7_5, A7_5 )-...
        PDE( TXY, lamb8, TX8, C8, A8, lamb8_7, TX8_7, C8_7, A8_7, lamb8_6, TX8_6, C8_6, A8_6 )-...
        PDE( TXY, lamb9, TX9, C9, A9, lamb9_8, TX9_8, C9_8, A9_8, lamb9_7, TX9_7, C9_7, A9_7 )-...
        PDE( TXY, lamb10, TX10, C10, A10, lamb10_9, TX10_9, C10_9, A10_9, lamb10_8, TX10_8, C10_8, A10_8 )-...
        PDE( TXY, lamb11, TX11, C11, A11, lamb11_10, TX11_10, C11_10, A11_10, lamb11_9, TX11_9, C11_9, A11_9 );

for i=1:Num

    if (TXY(i,1)-inx1(1,1))*(TXY(i,2)-inx1(1,2))*(TXY(i,2)-inx2(1,2))*(TXY(i,3)-inx1(1,3))*(TXY(i,3)-inx2(1,3)) == 0

        P(i,:)=  FAI(i,:);
        
        U(i)=boundcondition(TXY(i,:))-...
            (inner_test(TXY(i,:),lamb5,TX5,C5,A5)-2*inner_test(TXY(i,:),lamb5_4,TX5_4,C5_4,A5_4)+inner_test(TXY(i,:),lamb5_3,TX5_3,C5_3,A5_3))-...
            (inner_test(TXY(i,:),lamb6,TX6,C6,A6)-2*inner_test(TXY(i,:),lamb6_5,TX6_5,C6_5,A6_5)+inner_test(TXY(i,:),lamb6_4,TX6_4,C6_4,A6_4))-...
            (inner_test(TXY(i,:),lamb7,TX7,C7,A7)-2*inner_test(TXY(i,:),lamb7_6,TX7_6,C7_6,A7_6)+inner_test(TXY(i,:),lamb7_5,TX7_5,C7_5,A7_5))-...
            (inner_test(TXY(i,:),lamb8,TX8,C8,A8)-2*inner_test(TXY(i,:),lamb8_7,TX8_7,C8_7,A8_7)+inner_test(TXY(i,:),lamb8_6,TX8_6,C8_6,A8_6))-...
            (inner_test(TXY(i,:),lamb9,TX9,C9,A9)-2*inner_test(TXY(i,:),lamb9_8,TX9_8,C9_8,A9_8)+inner_test(TXY(i,:),lamb9_7,TX9_7,C9_7,A9_7))-...
            (inner_test(TXY(i,:),lamb10,TX10,C10,A10)-2*inner_test(TXY(i,:),lamb10_9,TX10_9,C10_9,A10_9)+inner_test(TXY(i,:),lamb10_8,TX10_8,C10_8,A10_8))-...
            (inner_test(TXY(i,:),lamb11,TX11,C11,A11)-2*inner_test(TXY(i,:),lamb11_10,TX11_10,C11_10,A11_10)+inner_test(TXY(i,:),lamb11_9,TX11_9,C11_9,A11_9));  % boundary condition
    end

end

% cond(P)
[F,J]=lu(P);
Jlamda=F\U;
lamb=J\Jlamda;

end



