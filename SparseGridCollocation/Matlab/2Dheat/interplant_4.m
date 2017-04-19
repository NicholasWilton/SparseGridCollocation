function [ lamb, TX, C, A, PU ] = interplant_4( b, d, inx1, inx2, coef, lamb5, TX5, C5, A5,...
    lamb5_4, TX5_4, C5_4, A5_4, lamb6, TX6, C6, A6,...
    lamb6_5, TX6_5, C6_5, A6_5, lamb7, TX7, C7, A7,...
    lamb7_6, TX7_6, C7_6, A7_6, lamb8, TX8, C8, A8,...
    lamb8_7, TX8_7, C8_7, A8_7 )

L=subnumber( b, d);

[ch,~]=size( L );

N=ones(ch,d);
for i=1:d
    N(:,i)=2.^L(:,i)+1;
end

parfor i = 1 : ch
    [ lamb{i}, TX{i} , C{i}, A{i}, PU{i} ]=...
        shapelambda2D_4( coef, inx1, inx2, N(i,:), lamb5, TX5, C5, A5,...
    lamb5_4, TX5_4, C5_4, A5_4, lamb6, TX6, C6, A6,...
    lamb6_5, TX6_5, C6_5, A6_5, lamb7, TX7, C7, A7,...
    lamb7_6, TX7_6, C7_6, A7_6, lamb8, TX8, C8, A8,...
    lamb8_7, TX8_7, C8_7, A8_7 );
end






end


