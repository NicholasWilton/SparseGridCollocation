function     [ lamb, TX , C, A ]=shapelambda2D(coef, tsec, r, sigma, T, E, inx1, inx2, N)

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
% This P is the PDE matrix determined specially for B-S
P = FAI_t + sigma^2*FAI_xx/2 + r*FAI_x - r*FAI;

for i=1:Num
%   choose out the nodes located on boundary
    if TX(i,2) == inx1 || TX(i,2) == inx2       
%   corresponding lines in PDE matrix P is replaced by interpolation lines
        P(i,:) = FAI(i,:);      
%   And the value in U is also replaced
        U(i)=max( 0, TX(i,2) - E*exp( -r * (T-TX(i,1)) ) );  % boundary condition
    end
%   choose out the nodes located on initial condition        
    if TX(i,1) == tsec
        P(i,:) = FAI(i,:);
        U(i) = PPP( TX(i,:) );
    end

end

% cond(P)
[F,J]=lu(P);
Jlamda=F\U;
lamb=J\Jlamda;

end

