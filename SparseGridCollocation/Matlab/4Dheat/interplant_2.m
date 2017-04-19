function [ lamb, TX, C, A, PU ] = interplant_2( b, d, inx1, inx2, coef, lamb7, TX7, C7, A7,...
    lamb7_6, TX7_6, C7_6, A7_6, lamb7_5, TX7_5, C7_5, A7_5, lamb7_4, TX7_4, C7_4, A7_4,...
    lamb8, TX8, C8, A8, lamb8_7, TX8_7, C8_7, A8_7, lamb8_6, TX8_6, C8_6, A8_6, lamb8_5, TX8_5, C8_5, A8_5)
L=subnumber( b, d);

[ch,~]=size( L );

N=ones(ch,d);
for i=1:d
    N(:,i)=2.^L(:,i)+1;
end

parfor i = 1 : ch
    [ lamb{i}, TX{i} , C{i}, A{i}, PU{i} ]=...
        shapelambda2D_2( coef, inx1, inx2, N(i,:), lamb7, TX7, C7, A7,...
    lamb7_6, TX7_6, C7_6, A7_6, lamb7_5, TX7_5, C7_5, A7_5, lamb7_4, TX7_4, C7_4, A7_4,...
    lamb8, TX8, C8, A8, lamb8_7, TX8_7, C8_7, A8_7, lamb8_6, TX8_6, C8_6, A8_6, lamb8_5, TX8_5, C8_5, A8_5);
end

end
