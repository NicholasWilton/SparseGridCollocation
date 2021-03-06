function [ lamb, TX, C, A, PU ] = interplant_9( coef, tsec, b, d, inx1, inx2, r, sigma, T, E, lamb2, TX2, C2, A2,lamb3, TX3, C3, A3,...
    lamb_3, TX_3, C_3, A_3,lamb4, TX4, C4, A4,lamb_4, TX_4, C_4, A_4,lamb5, TX5, C5, A5,...
    lamb_5, TX_5, C_5, A_5,lamb6, TX6, C6, A6,lamb_6, TX_6, C_6, A_6,lamb7, TX7, C7, A7,...
    lamb_7, TX_7, C_7, A_7,lamb8, TX8, C8, A8,lamb_8, TX_8, C_8, A_8,lamb9, TX9, C9, A9,...
    lamb_9, TX_9, C_9, A_9,lamb10, TX10, C10, A10,...
    lamb_10, TX_10, C_10, A_10,lamb11, TX11, C11, A11)

L=subnumber( b, d);

[ch,~]=size( L );

N=ones(ch,d);
for i=1:d
    N(:,i)=2.^L(:,i)+1;
end

parfor i = 1 : ch
    [ lamb{i}, TX{i} , C{i}, A{i}, PU{i} ]=...
        shapelambda2D_9(coef,tsec,r,sigma,T,E,inx1,inx2,N(i,:),lamb2, TX2, C2, A2,lamb3, TX3, C3, A3,...
        lamb_3, TX_3, C_3, A_3,lamb4, TX4, C4, A4, lamb_4, TX_4, C_4, A_4,lamb5, TX5, C5, A5,...
        lamb_5, TX_5, C_5, A_5,lamb6, TX6, C6, A6,lamb_6, TX_6, C_6, A_6,lamb7, TX7, C7, A7,...
        lamb_7, TX_7, C_7, A_7,lamb8, TX8, C8, A8,lamb_8, TX_8, C_8, A_8,lamb9, TX9, C9, A9,...
        lamb_9, TX_9, C_9, A_9,lamb10, TX10, C10, A10, lamb_10, TX_10, C_10, A_10,lamb11, TX11, C11, A11);
end






end



