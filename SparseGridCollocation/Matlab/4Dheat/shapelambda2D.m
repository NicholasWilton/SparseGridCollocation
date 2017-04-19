function     [ lamb, TXYZ , C, A ]=shapelambda2D(coef,inx1,inx2,N)
% cha stands for the length in every dimension
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
% This P is the PDE matrix determined specially for different PDE, here it is heat PDE
P=FAI_t - FAI_xx - FAI_yy - FAI_zz;

for i=1:Num
%   choose out the nodes located on initial and boundary
    if (TXYZ(i,1)-inx1(1,1))*(TXYZ(i,2)-inx1(1,2))*(TXYZ(i,2)-inx2(1,2))*...
                             (TXYZ(i,3)-inx1(1,3))*(TXYZ(i,3)-inx2(1,3))*...
                             (TXYZ(i,4)-inx1(1,4))*(TXYZ(i,4)-inx2(1,4)) == 0
%   corresponding lines in PDE matrix P is replaced by interpolation lines
        P(i,:)=  FAI(i,:);
%   And the value in U is also replaced           
        U(i)=boundcondition(TXYZ(i,:));  % boundary condition
    end

end

% cond(P)
[F,J]=lu(P);
Jlamda=F\U;
lamb=J\Jlamda;

end

