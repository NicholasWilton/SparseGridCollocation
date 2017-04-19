function [ lamb, TX, C, A ] = interplant( coef, tsec, b, d, inx1, inx2, r, sigma, T, E )
% b = level n + dimension d - 1;
    
L=subnumber( b, d);

[ch,~]=size( L );

N=ones(ch,d);
for i=1:d
    N(:,i)=2.^L(:,i)+1;
end
% calculate information on every sub grid
%parfor i = 1 : ch
for i = 1 : ch
    %if i <= ch
    [ lamb{i}, TX{i} , C{i}, A{i} ]=...
        shapelambda2D(coef,tsec,r,sigma,T,E,inx1,inx2,N(i,:));
    %end
end

end

