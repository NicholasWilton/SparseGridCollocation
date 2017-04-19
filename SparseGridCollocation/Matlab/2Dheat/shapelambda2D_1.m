function     [ lamb, TX , C, A , U ]=shapelambda2D_1( coef, inx1, inx2, N, lamb5, TX5, C5, A5,...
    lamb5_4, TX5_4, C5_4, A5_4 )

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
% re-construct the PDE
U = U - PDE( TX,lamb5, TX5, C5, A5, lamb5_4, TX5_4, C5_4, A5_4 );

for i=1:Num

    if (TX(i,1)-inx1(1,1))*(TX(i,2)-inx1(1,2))*(TX(i,2)-inx2(1,2))  == 0

        P(i,:)=  FAI(i,:);
%       re-construct the initial and boundary conditions
        U(i)=boundcondition(TX(i,:))-...
            (inner_test(TX(i,:),lamb5,TX5,C5,A5)-inner_test(TX(i,:),lamb5_4,TX5_4,C5_4,A5_4));
    end

end

% cond(P)
[F,J]=lu(P);
Jlamda=F\U;
lamb=J\Jlamda;

end



