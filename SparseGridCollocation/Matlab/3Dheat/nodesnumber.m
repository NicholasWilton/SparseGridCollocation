function [ output ] = nodesnumber( n, d )
% To calculate the number of nodes in one sparse grid at level n-2 and
% here dimension d must be 3 in 3D problem
for i=n-2:n
    L=subnumber( i, d);
    [ch,~]=size( L );
    N=ones(ch,d);
    for j=1:d
        N(:,j)=2.^L(:,j)+1;
    end
    P=prod(N,2);
    Num(i-(n-3))=sum(P);
end
output=Num*[1;-2;1];

