function [ lamb, TX, C, A, PU ] = interplant_3( coef, tsec, b, d, inx1, inx2, r, sigma, T, E, lamb2, TX2, C2, A2,lamb3, TX3, C3, A3,...
    lamb_3, TX_3, C_3, A_3,lamb4, TX4, C4, A4,lamb_4, TX_4, C_4, A_4,lamb5, TX5, C5, A5 )

L=subnumber( b, d);

[ch,~]=size( L );

N=ones(ch,d);
for i=1:d
    N(:,i)=2.^L(:,i)+1;
end

parfor i = 1 : ch
    [ lamb{i}, TX{i} , C{i}, A{i}, PU{i} ]=...
        shapelambda2D_3(coef,tsec,r,sigma,T,E,inx1,inx2,N(i,:),lamb2, TX2, C2, A2,lamb3, TX3, C3, A3,...
        lamb_3, TX_3, C_3, A_3,lamb4, TX4, C4, A4, lamb_4, TX_4, C_4, A_4,lamb5, TX5, C5, A5 );
end






end
