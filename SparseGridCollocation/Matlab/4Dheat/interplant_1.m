function [ lamb, TX, C, A, PU ] = interplant_1( b, d, inx1, inx2, coef, lamb7, TX7, C7, A7,...
    lamb7_6, TX7_6, C7_6, A7_6, lamb7_5, TX7_5, C7_5, A7_5, lamb7_4, TX7_4, C7_4, A7_4)

L=subnumber( b, d);

[ch,~]=size( L );

N=ones(ch,d);
for i=1:d
    N(:,i)=2.^L(:,i)+1;
end

parfor i = 1 : ch
    [ lamb{i}, TX{i} , C{i}, A{i}, PU{i} ]=...
        shapelambda2D_1( coef, inx1, inx2, N(i,:), lamb7, TX7, C7, A7,...
    lamb7_6, TX7_6, C7_6, A7_6, lamb7_5, TX7_5, C7_5, A7_5, lamb7_4, TX7_4, C7_4, A7_4);
end

end


