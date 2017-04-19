function [ lamb, TX, C, A, PU ] = interplant_2( b, d, inx1, inx2, coef, lamb5, TX5, C5, A5,...
    lamb5_4, TX5_4, C5_4, A5_4, lamb5_3, TX5_3, C5_3, A5_3, lamb6, TX6, C6, A6,...
    lamb6_5, TX6_5, C6_5, A6_5, lamb6_4, TX6_4, C6_4, A6_4)
L=subnumber( b, d);

[ch,~]=size( L );

N=ones(ch,d);
for i=1:d
    N(:,i)=2.^L(:,i)+1;
end

parfor i = 1 : ch
    [ lamb{i}, TX{i} , C{i}, A{i}, PU{i} ]=...
        shapelambda2D_2( coef, inx1, inx2, N(i,:), lamb5, TX5, C5, A5,...
    lamb5_4, TX5_4, C5_4, A5_4, lamb5_3, TX5_3, C5_3, A5_3, lamb6, TX6, C6, A6,...
    lamb6_5, TX6_5, C6_5, A6_5, lamb6_4, TX6_4, C6_4, A6_4);
end






end
