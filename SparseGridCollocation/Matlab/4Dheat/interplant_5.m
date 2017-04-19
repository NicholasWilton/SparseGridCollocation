function [ lamb, TX, C, A, PU ] = interplant_5( b, d, inx1, inx2, coef, lamb7, TX7, C7, A7,...
    lamb7_6, TX7_6, C7_6, A7_6, lamb7_5, TX7_5, C7_5, A7_5, lamb7_4, TX7_4, C7_4, A7_4,...
    lamb8, TX8, C8, A8, lamb8_7, TX8_7, C8_7, A8_7, lamb8_6, TX8_6, C8_6, A8_6, lamb8_5, TX8_5, C8_5, A8_5,...
    lamb9, TX9, C9, A9, lamb9_8, TX9_8, C9_8, A9_8, lamb9_7, TX9_7, C9_7, A9_7, lamb9_6, TX9_6, C9_6, A9_6,...
    lamb10, TX10, C10, A10, lamb10_9, TX10_9, C10_9, A10_9, lamb10_8, TX10_8, C10_8, A10_8, lamb10_7, TX10_7, C10_7, A10_7,...
    lamb11, TX11, C11, A11, lamb11_10, TX11_10, C11_10, A11_10, lamb11_9, TX11_9, C11_9, A11_9, lamb11_8, TX11_8, C11_8, A11_8 )

L=subnumber( b, d);

[ch,~]=size( L );

N=ones(ch,d);
for i=1:d
    N(:,i)=2.^L(:,i)+1;
end

parfor i = 1 : ch
    [ lamb{i}, TX{i} , C{i}, A{i}, PU{i} ]=...
        shapelambda2D_5( coef, inx1, inx2, N(i,:), lamb7, TX7, C7, A7,...
    lamb7_6, TX7_6, C7_6, A7_6, lamb7_5, TX7_5, C7_5, A7_5, lamb7_4, TX7_4, C7_4, A7_4,...
    lamb8, TX8, C8, A8, lamb8_7, TX8_7, C8_7, A8_7, lamb8_6, TX8_6, C8_6, A8_6, lamb8_5, TX8_5, C8_5, A8_5,...
    lamb9, TX9, C9, A9, lamb9_8, TX9_8, C9_8, A9_8, lamb9_7, TX9_7, C9_7, A9_7, lamb9_6, TX9_6, C9_6, A9_6,...
    lamb10, TX10, C10, A10, lamb10_9, TX10_9, C10_9, A10_9, lamb10_8, TX10_8, C10_8, A10_8, lamb10_7, TX10_7, C10_7, A10_7,...
    lamb11, TX11, C11, A11, lamb11_10, TX11_10, C11_10, A11_10, lamb11_9, TX11_9, C11_9, A11_9, lamb11_8, TX11_8, C11_8, A11_8);
end

end


