function     [ lamb, TXY , C, A , U ]=shapelambda2D_1( coef, inx1, inx2, N, lamb5, TX5, C5, A5,...
    lamb5_4, TX5_4, C5_4, A5_4, lamb5_3, TX5_3, C5_3, A5_3)

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

U = U - PDE( TXY,lamb5, TX5, C5, A5,...
    lamb5_4, TX5_4, C5_4, A5_4, lamb5_3, TX5_3, C5_3, A5_3 );

for i=1:Num

    if (TXY(i,1)-inx1(1,1))*(TXY(i,2)-inx1(1,2))*(TXY(i,2)-inx2(1,2))*(TXY(i,3)-inx1(1,3))*(TXY(i,3)-inx2(1,3)) == 0

        P(i,:)=  FAI(i,:);
%       re-construct the initial and boundary conditions        
        U(i)=boundcondition(TXY(i,:))-...
            (inner_test(TXY(i,:),lamb5,TX5,C5,A5)-2*inner_test(TXY(i,:),lamb5_4,TX5_4,C5_4,A5_4)+inner_test(TXY(i,:),lamb5_3,TX5_3,C5_3,A5_3));  % boundary condition
    end

end

% cond(P)
[F,J]=lu(P);
Jlamda=F\U;
lamb=J\Jlamda;

end



