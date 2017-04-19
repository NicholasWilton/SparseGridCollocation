function     [ lamb, TX , C, A ]=shapelambda2D(coef,inx1,inx2,N)
% cha stands for the length in every dimension
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
% This P is the PDE matrix determined specially for different PDE, here it is heat PDE
P = FAI_t - FAI_xx;

for i=1:Num
%   choose out the nodes located on initial and boundary
    if (TX(i,1)-inx1(1,1))*(TX(i,2)-inx1(1,2))*(TX(i,2)-inx2(1,2)) == 0
%   corresponding lines in PDE matrix P is replaced by interpolation lines
        P(i,:)=  FAI(i,:);
%   And the value in U is also replaced
        U(i)=boundcondition(TX(i,:));  % boundary condition
    end

end

% cond(P)
[F,J]=lu(P);
Jlamda=F\U;
lamb=J\Jlamda;

end

