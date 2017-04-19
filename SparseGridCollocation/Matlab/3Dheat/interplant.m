function [ lamb, TX, C, A ] = interplant( b, d, inx1, inx2, coef )
% b = level n + dimension d - 1;
L=subnumber( b, d);

[ch,~]=size( L );

N=ones(ch,d);
for i=1:d
    N(:,i)=2.^L(:,i)+1;
end
% calculate information on every sub grid
parfor i = 1 : ch
    [ lamb{i}, TX{i} , C{i}, A{i} ]=...
        shapelambda2D(coef,inx1,inx2,N(i,:));
end

end

