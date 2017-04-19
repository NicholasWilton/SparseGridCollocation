function [ output ] = nodesnumber( n, d )
% To calculate the number of nodes in one sparse grid at level n and
% here dimension d must be 4 in 4D problem
for i=n:n+3
    L=subnumber( i, d);
    [ch,~]=size( L );
    N=ones(ch,d);
    for j=1:d
        N(:,j)=2.^L(:,j)+1;
    end
    P=prod(N,2);
    Num(i-(n-1))=sum(P);
end
output=Num*[-1;3;-3;1];

