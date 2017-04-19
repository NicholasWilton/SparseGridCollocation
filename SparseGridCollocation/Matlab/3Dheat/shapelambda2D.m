function     [ lamb, TXY , C, A ]=shapelambda2D(coef,inx1,inx2,N)
% cha stands for the length in every dimension
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
% This P is the PDE matrix determined specially for different PDE, here it is heat PDE
P=FAI_t - FAI_xx - FAI_yy;

for i=1:Num
%   choose out the nodes located on initial and boundary
    if (TXY(i,1)-inx1(1,1))*(TXY(i,2)-inx1(1,2))*(TXY(i,2)-inx2(1,2))*(TXY(i,3)-inx1(1,3))*(TXY(i,3)-inx2(1,3)) == 0
%   corresponding lines in PDE matrix P is replaced by interpolation lines
        P(i,:)=  FAI(i,:);
%   And the value in U is also replaced        
        U(i)=boundcondition(TXY(i,:));  % boundary condition
    end

end

% cond(P)
[F,J]=lu(P);
Jlamda=F\U;
lamb=J\Jlamda;

end

