function [ output ] = nodesnumber( n, d )
% To calculate the number of nodes in one sparse grid at level n-1 and
% here dimension d must be 2 in 2D problem
if n >= 2*d -1
   for i=n-(d-1):n
       L=subnumber( i, d);
       [ch,~]=size( L );
       N=ones(ch,d);
       for j=1:d
           N(:,j)=2.^L(:,j)+1;
       end
       P=prod(N,2);
       Num(i-(n-d))=sum(P);
   end
output=Num*[-1;1];
end

