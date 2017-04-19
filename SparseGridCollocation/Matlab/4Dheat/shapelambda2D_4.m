function     [ lamb, TXYZ , C, A, U ]=shapelambda2D_4( coef, inx1, inx2, N, lamb7, TX7, C7, A7,...
    lamb7_6, TX7_6, C7_6, A7_6, lamb7_5, TX7_5, C7_5, A7_5, lamb7_4, TX7_4, C7_4, A7_4,...
    lamb8, TX8, C8, A8, lamb8_7, TX8_7, C8_7, A8_7, lamb8_6, TX8_6, C8_6, A8_6, lamb8_5, TX8_5, C8_5, A8_5,...
    lamb9, TX9, C9, A9, lamb9_8, TX9_8, C9_8, A9_8, lamb9_7, TX9_7, C9_7, A9_7, lamb9_6, TX9_6, C9_6, A9_6,...
    lamb10, TX10, C10, A10, lamb10_9, TX10_9, C10_9, A10_9, lamb10_8, TX10_8, C10_8, A10_8, lamb10_7, TX10_7, C10_7, A10_7)

cha=inx2-inx1;
Num=prod(N);
t=linspace(inx1(1,1),inx2(1,1),N(1,1));
x=linspace(inx1(1,2),inx2(1,2),N(1,2));
y=linspace(inx1(1,3),inx2(1,3),N(1,3));
z=linspace(inx1(1,4),inx2(1,4),N(1,4));

C=coef*cha;
A=N-1;

[V1, V2, V3, V4] = ndgrid(t,x,y,z);
V1 = reshape(V1,[],1);
V2 = reshape(V2,[],1);
V3 = reshape(V3,[],1);
V4 = reshape(V4,[],1);
TXYZ = [V1 V2 V3 V4];

U=iniPDE(TXYZ);

[ FAI, FAI_t, FAI_xx, FAI_yy, FAI_zz ] = mq4d( TXYZ, TXYZ, A, C );

P=FAI_t - FAI_xx - FAI_yy - FAI_zz;

U = U - PDE( TXYZ, lamb7, TX7, C7, A7,...
    lamb7_6, TX7_6, C7_6, A7_6, lamb7_5, TX7_5, C7_5, A7_5, lamb7_4, TX7_4, C7_4, A7_4)-...
    PDE( TXYZ, lamb8, TX8, C8, A8, lamb8_7, TX8_7, C8_7, A8_7, lamb8_6, TX8_6, C8_6, A8_6, lamb8_5, TX8_5, C8_5, A8_5 )-...
    PDE( TXYZ, lamb9, TX9, C9, A9, lamb9_8, TX9_8, C9_8, A9_8, lamb9_7, TX9_7, C9_7, A9_7, lamb9_6, TX9_6, C9_6, A9_6 )-...
    PDE( TXYZ, lamb10, TX10, C10, A10, lamb10_9, TX10_9, C10_9, A10_9, lamb10_8, TX10_8, C10_8, A10_8, lamb10_7, TX10_7, C10_7, A10_7);

for i=1:Num

   if (TXYZ(i,1)-inx1(1,1))*(TXYZ(i,2)-inx1(1,2))*(TXYZ(i,2)-inx2(1,2))*...
                            (TXYZ(i,3)-inx1(1,3))*(TXYZ(i,3)-inx2(1,3))*...
                            (TXYZ(i,4)-inx1(1,4))*(TXYZ(i,4)-inx2(1,4)) == 0
        P(i,:)=  FAI(i,:);
        
        U(i)=boundcondition(TXYZ(i,:))-...
            ( inner_test( TXYZ(i,:),lamb7,TX7,C7,A7 ) - 3*inner_test( TXYZ(i,:),lamb7_6,TX7_6,C7_6,A7_6 ) +...
            3*inner_test( TXYZ(i,:),lamb7_5,TX7_5,C7_5,A7_5 ) - inner_test( TXYZ(i,:),lamb7_4,TX7_4,C7_4,A7_4 )) -...
            ( inner_test( TXYZ(i,:),lamb8,TX8,C8,A8 ) - 3*inner_test( TXYZ(i,:),lamb8_7,TX8_7,C8_7,A8_7 ) +...
            3*inner_test( TXYZ(i,:),lamb8_6,TX8_6,C8_6,A8_6 ) - inner_test( TXYZ(i,:),lamb8_5,TX8_5,C8_5,A8_5 )) -...
            ( inner_test( TXYZ(i,:),lamb9,TX9,C9,A9 ) - 3*inner_test( TXYZ(i,:),lamb9_8,TX9_8,C9_8,A9_8 ) +...
            3*inner_test( TXYZ(i,:),lamb9_7,TX9_7,C9_7,A9_7 ) - inner_test( TXYZ(i,:),lamb9_6,TX9_6,C9_6,A9_6 ))-...
            ( inner_test( TXYZ(i,:),lamb10,TX10,C10,A10 ) - 3*inner_test( TXYZ(i,:),lamb10_9,TX10_9,C10_9,A10_9 ) +...
            3*inner_test( TXYZ(i,:),lamb10_8,TX10_8,C10_8,A10_8 ) - inner_test( TXYZ(i,:),lamb10_7,TX10_7,C10_7,A10_7 ));  % boundary condition
    end

end

% cond(P)
[F,J]=lu(P);
Jlamda=F\U;
lamb=J\Jlamda;

end
