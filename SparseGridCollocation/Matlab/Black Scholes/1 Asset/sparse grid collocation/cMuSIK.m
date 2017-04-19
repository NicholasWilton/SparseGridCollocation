% multilevel sparse grid collocation for 2D BS ( time * 1D stock dimension )
load('Smoothinitial.mat', 'Tdone')

E=100; % strike price
r=0.03; % interest rate
sigma=0.15; 
T=1; % Maturity
inx1=0; % stock price S belongs to [inx1 inx2]
inx2=3*E;
tsec=T-Tdone; % Initial time boundary for sparse grid 
d=2; % dimension
coef = 2; % coef stands for the connection constant number

ch=10000;
x=linspace( inx1, inx2, ch);
t = zeros(ch,1);
TX = [t x']; % testing nodes
% na, nb = level n + dimension d - 1
na=3;
nb=2;
tic
% Level 2 ....lamb stands for \lambda the coefficients, TX stands for nodes
% C stands for shape parater, A stands for scale parameter
[ lamb3, TX3, C3, A3 ] = interplant( coef, tsec, na, d, inx1, inx2, r, sigma, T, E );

[ lamb2, TX2, C2, A2 ] = interplant( coef, tsec, nb, d, inx1, inx2, r, sigma, T, E );
ttt(1) = toc;
% .............
% Level 3 .... multilevel method has to use all previous information
[ lamb4, TX4, C4, A4, PU4 ] = interplant_1( coef, tsec, na+1, d, inx1, inx2, r, sigma, T, E, lamb2, TX2, C2, A2,lamb3, TX3, C3, A3 );

[ lamb_3, TX_3, C_3, A_3, PU_3 ] = interplant_1( coef, tsec, nb+1, d, inx1, inx2, r, sigma, T, E, lamb2, TX2, C2, A2,lamb3, TX3, C3, A3 );
ttt(2) = toc;
% .............
% Level 4 .... higher level needs more information
[ lamb5, TX5, C5, A5, PU5 ] = interplant_2( coef, tsec, na+2, d, inx1, inx2, r, sigma, T, E, lamb2, TX2, C2, A2,lamb3, TX3, C3, A3,...
    lamb_3, TX_3, C_3, A_3,lamb4, TX4, C4, A4);

[ lamb_4, TX_4, C_4, A_4, PU_4 ] = interplant_2( coef, tsec, nb+2, d, inx1, inx2, r, sigma, T, E, lamb2, TX2, C2, A2,lamb3, TX3, C3, A3,...
    lamb_3, TX_3, C_3, A_3,lamb4, TX4, C4, A4);
ttt(3) = toc;
% ............
% Level 5
[ lamb6, TX6, C6, A6, PU6 ] = interplant_3( coef, tsec, na+3, d, inx1, inx2, r, sigma, T, E, lamb2, TX2, C2, A2,lamb3, TX3, C3, A3,...
    lamb_3, TX_3, C_3, A_3,lamb4, TX4, C4, A4,lamb_4, TX_4, C_4, A_4,lamb5, TX5, C5, A5 );

[ lamb_5, TX_5, C_5, A_5, PU_5 ] = interplant_3( coef, tsec, nb+3, d, inx1, inx2, r, sigma, T, E, lamb2, TX2, C2, A2,lamb3, TX3, C3, A3,...
    lamb_3, TX_3, C_3, A_3,lamb4, TX4, C4, A4,lamb_4, TX_4, C_4, A_4,lamb5, TX5, C5, A5 );
ttt(4) = toc;
% ............
% Level 6
[ lamb7, TX7, C7, A7, PU7 ] = interplant_4( coef, tsec, na+4, d, inx1, inx2, r, sigma, T, E, lamb2, TX2, C2, A2,lamb3, TX3, C3, A3,...
    lamb_3, TX_3, C_3, A_3,lamb4, TX4, C4, A4,lamb_4, TX_4, C_4, A_4,lamb5, TX5, C5, A5,...
    lamb_5, TX_5, C_5, A_5,lamb6, TX6, C6, A6);

[ lamb_6, TX_6, C_6, A_6, PU_6 ] = interplant_4( coef, tsec, nb+4, d, inx1, inx2, r, sigma, T, E, lamb2, TX2, C2, A2,lamb3, TX3, C3, A3,...
    lamb_3, TX_3, C_3, A_3,lamb4, TX4, C4, A4,lamb_4, TX_4, C_4, A_4,lamb5, TX5, C5, A5,...
    lamb_5, TX_5, C_5, A_5,lamb6, TX6, C6, A6);
ttt(5) = toc;
% ............
% Level 7
[ lamb8, TX8, C8, A8 , PU8 ] = interplant_5( coef, tsec, na+5, d, inx1, inx2, r, sigma, T, E, lamb2, TX2, C2, A2,lamb3, TX3, C3, A3,...
    lamb_3, TX_3, C_3, A_3,lamb4, TX4, C4, A4,lamb_4, TX_4, C_4, A_4,lamb5, TX5, C5, A5,...
    lamb_5, TX_5, C_5, A_5,lamb6, TX6, C6, A6,lamb_6, TX_6, C_6, A_6,lamb7, TX7, C7, A7 );

[ lamb_7, TX_7, C_7, A_7 , PU_7] = interplant_5( coef, tsec, nb+5, d, inx1, inx2, r, sigma, T, E, lamb2, TX2, C2, A2,lamb3, TX3, C3, A3,...
    lamb_3, TX_3, C_3, A_3,lamb4, TX4, C4, A4,lamb_4, TX_4, C_4, A_4,lamb5, TX5, C5, A5,...
    lamb_5, TX_5, C_5, A_5,lamb6, TX6, C6, A6,lamb_6, TX_6, C_6, A_6,lamb7, TX7, C7, A7 );
ttt(6) = toc;
% ............
% Level 8
[ lamb9, TX9, C9, A9, PU9 ] = interplant_6( coef, tsec, na+6, d, inx1, inx2, r, sigma, T, E, lamb2, TX2, C2, A2,lamb3, TX3, C3, A3,...
    lamb_3, TX_3, C_3, A_3,lamb4, TX4, C4, A4,lamb_4, TX_4, C_4, A_4,lamb5, TX5, C5, A5,...
    lamb_5, TX_5, C_5, A_5,lamb6, TX6, C6, A6,lamb_6, TX_6, C_6, A_6,lamb7, TX7, C7, A7,...
    lamb_7, TX_7, C_7, A_7,lamb8, TX8, C8, A8);

[ lamb_8, TX_8, C_8, A_8 , PU_8] = interplant_6( coef, tsec, nb+6, d, inx1, inx2, r, sigma, T, E, lamb2, TX2, C2, A2,lamb3, TX3, C3, A3,...
    lamb_3, TX_3, C_3, A_3,lamb4, TX4, C4, A4,lamb_4, TX_4, C_4, A_4,lamb5, TX5, C5, A5,...
    lamb_5, TX_5, C_5, A_5,lamb6, TX6, C6, A6,lamb_6, TX_6, C_6, A_6,lamb7, TX7, C7, A7,...
    lamb_7, TX_7, C_7, A_7,lamb8, TX8, C8, A8);
ttt(7) = toc;
% ............
% Level 9
[ lamb10, TX10, C10, A10, PU10 ] = interplant_7( coef, tsec, na+7, d, inx1, inx2, r, sigma, T, E, lamb2, TX2, C2, A2,lamb3, TX3, C3, A3,...
    lamb_3, TX_3, C_3, A_3,lamb4, TX4, C4, A4,lamb_4, TX_4, C_4, A_4,lamb5, TX5, C5, A5,...
    lamb_5, TX_5, C_5, A_5,lamb6, TX6, C6, A6,lamb_6, TX_6, C_6, A_6,lamb7, TX7, C7, A7,...
    lamb_7, TX_7, C_7, A_7,lamb8, TX8, C8, A8,lamb_8, TX_8, C_8, A_8,lamb9, TX9, C9, A9);

[ lamb_9, TX_9, C_9, A_9, PU_9 ] = interplant_7( coef, tsec, nb+7, d, inx1, inx2, r, sigma, T, E, lamb2, TX2, C2, A2,lamb3, TX3, C3, A3,...
    lamb_3, TX_3, C_3, A_3,lamb4, TX4, C4, A4,lamb_4, TX_4, C_4, A_4,lamb5, TX5, C5, A5,...
    lamb_5, TX_5, C_5, A_5,lamb6, TX6, C6, A6,lamb_6, TX_6, C_6, A_6,lamb7, TX7, C7, A7,...
    lamb_7, TX_7, C_7, A_7,lamb8, TX8, C8, A8,lamb_8, TX_8, C_8, A_8,lamb9, TX9, C9, A9);
ttt(8) = toc;
% ............
% Level 10
[ lamb11, TX11, C11, A11, PU11 ] = interplant_8( coef, tsec, na+8, d, inx1, inx2, r, sigma, T, E, lamb2, TX2, C2, A2,lamb3, TX3, C3, A3,...
    lamb_3, TX_3, C_3, A_3,lamb4, TX4, C4, A4,lamb_4, TX_4, C_4, A_4,lamb5, TX5, C5, A5,...
    lamb_5, TX_5, C_5, A_5,lamb6, TX6, C6, A6,lamb_6, TX_6, C_6, A_6,lamb7, TX7, C7, A7,...
    lamb_7, TX_7, C_7, A_7,lamb8, TX8, C8, A8,lamb_8, TX_8, C_8, A_8,lamb9, TX9, C9, A9,...
    lamb_9, TX_9, C_9, A_9,lamb10, TX10, C10, A10);

[ lamb_10, TX_10, C_10, A_10, PU_10 ] = interplant_8( coef, tsec, nb+8, d, inx1, inx2, r, sigma, T, E, lamb2, TX2, C2, A2,lamb3, TX3, C3, A3,...
    lamb_3, TX_3, C_3, A_3,lamb4, TX4, C4, A4,lamb_4, TX_4, C_4, A_4,lamb5, TX5, C5, A5,...
    lamb_5, TX_5, C_5, A_5,lamb6, TX6, C6, A6,lamb_6, TX_6, C_6, A_6,lamb7, TX7, C7, A7,...
    lamb_7, TX_7, C_7, A_7,lamb8, TX8, C8, A8,lamb_8, TX_8, C_8, A_8,lamb9, TX9, C9, A9,...
    lamb_9, TX_9, C_9, A_9,lamb10, TX10, C10, A10);
ttt(9) = toc;
% ............
% Level 11
% [ lamb12, TX12, C12, A12 ] = interplant_9( coef, tsec, na+9, d, inx1, inx2, r, sigma, T, E, lamb2, TX2, C2, A2,lamb3, TX3, C3, A3,...
%     lamb_3, TX_3, C_3, A_3,lamb4, TX4, C4, A4,lamb_4, TX_4, C_4, A_4,lamb5, TX5, C5, A5,...
%     lamb_5, TX_5, C_5, A_5,lamb6, TX6, C6, A6,lamb_6, TX_6, C_6, A_6,lamb7, TX7, C7, A7,...
%     lamb_7, TX_7, C_7, A_7,lamb8, TX8, C8, A8,lamb_8, TX_8, C_8, A_8,lamb9, TX9, C9, A9,...
%     lamb_9, TX_9, C_9, A_9,lamb10, TX10, C10, A10, lamb_10, TX_10, C_10, A_10,lamb11, TX11, C11, A11);
% 
% [ lamb_11, TX_11, C_11, A_11 ] = interplant_9( coef, tsec, nb+9, d, inx1, inx2, r, sigma, T, E, lamb2, TX2, C2, A2,lamb3, TX3, C3, A3,...
%     lamb_3, TX_3, C_3, A_3,lamb4, TX4, C4, A4,lamb_4, TX_4, C_4, A_4,lamb5, TX5, C5, A5,...
%     lamb_5, TX_5, C_5, A_5,lamb6, TX6, C6, A6,lamb_6, TX_6, C_6, A_6,lamb7, TX7, C7, A7,...
%     lamb_7, TX_7, C_7, A_7,lamb8, TX8, C8, A8,lamb_8, TX_8, C_8, A_8,lamb9, TX9, C9, A9,...
%     lamb_9, TX_9, C_9, A_9,lamb10, TX10, C10, A10, lamb_10, TX_10, C_10, A_10,lamb11, TX11, C11, A11);
% ttt(10) = toc;
% ............
% Level 12
% [ lamb13, TX13, C13, A13 ] = interplant_10( coef, tsec, na+10, d, inx1, inx2, r, sigma, T, E, lamb2, TX2, C2, A2,lamb3, TX3, C3, A3,...
%     lamb_3, TX_3, C_3, A_3,lamb4, TX4, C4, A4,lamb_4, TX_4, C_4, A_4,lamb5, TX5, C5, A5,...
%     lamb_5, TX_5, C_5, A_5,lamb6, TX6, C6, A6,lamb_6, TX_6, C_6, A_6,lamb7, TX7, C7, A7,...
%     lamb_7, TX_7, C_7, A_7,lamb8, TX8, C8, A8,lamb_8, TX_8, C_8, A_8,lamb9, TX9, C9, A9,...
%     lamb_9, TX_9, C_9, A_9,lamb10, TX10, C10, A10,...
%     lamb_10, TX_10, C_10, A_10,lamb11, TX11, C11, A11,...
%     lamb_11, TX_11, C_11, A_11,lamb12, TX12, C12, A12);
% [ lamb_12, TX_12, C_12, A_12 ] = interplant_10( coef, tsec, nb+10, d, inx1, inx2, r, sigma, T, E, lamb2, TX2, C2, A2,lamb3, TX3, C3, A3,...
%     lamb_3, TX_3, C_3, A_3,lamb4, TX4, C4, A4,lamb_4, TX_4, C_4, A_4,lamb5, TX5, C5, A5,...
%     lamb_5, TX_5, C_5, A_5,lamb6, TX6, C6, A6,lamb_6, TX_6, C_6, A_6,lamb7, TX7, C7, A7,...
%     lamb_7, TX_7, C_7, A_7,lamb8, TX8, C8, A8,lamb_8, TX_8, C_8, A_8,lamb9, TX9, C9, A9,...
%     lamb_9, TX_9, C_9, A_9,lamb10, TX10, C10, A10,...
%     lamb_10, TX_10, C_10, A_10,lamb11, TX11, C11, A11,...
%     lamb_11, TX_11, C_11, A_11,lamb12, TX12, C12, A12);
% ttt(11)=toc;
% ............
% Level 13
% [ lamb14, TX14, C14, A14 ] = interplant_11( coef, tsec, na+11, d, inx1, inx2, r, sigma, T, E, lamb2, TX2, C2, A2,lamb3, TX3, C3, A3,...
%     lamb_3, TX_3, C_3, A_3,lamb4, TX4, C4, A4,lamb_4, TX_4, C_4, A_4,lamb5, TX5, C5, A5,...
%     lamb_5, TX_5, C_5, A_5,lamb6, TX6, C6, A6,lamb_6, TX_6, C_6, A_6,lamb7, TX7, C7, A7,...
%     lamb_7, TX_7, C_7, A_7,lamb8, TX8, C8, A8,lamb_8, TX_8, C_8, A_8,lamb9, TX9, C9, A9,...
%     lamb_9, TX_9, C_9, A_9,lamb10, TX10, C10, A10,...
%     lamb_10, TX_10, C_10, A_10,lamb11, TX11, C11, A11,...
%     lamb_11, TX_11, C_11, A_11,lamb12, TX12, C12, A12,...
%     lamb_12, TX_12, C_12, A_12,lamb13, TX13, C13, A13);
% [ lamb_13, TX_13, C_13, A_13 ] = interplant_11( coef, tsec, nb+11, d, inx1, inx2, r, sigma, T, E, lamb2, TX2, C2, A2,lamb3, TX3, C3, A3,...
%     lamb_3, TX_3, C_3, A_3,lamb4, TX4, C4, A4,lamb_4, TX_4, C_4, A_4,lamb5, TX5, C5, A5,...
%     lamb_5, TX_5, C_5, A_5,lamb6, TX6, C6, A6,lamb_6, TX_6, C_6, A_6,lamb7, TX7, C7, A7,...
%     lamb_7, TX_7, C_7, A_7,lamb8, TX8, C8, A8,lamb_8, TX_8, C_8, A_8,lamb9, TX9, C9, A9,...
%     lamb_9, TX_9, C_9, A_9,lamb10, TX10, C10, A10,...
%     lamb_10, TX_10, C_10, A_10,lamb11, TX11, C11, A11,...
%     lamb_11, TX_11, C_11, A_11,lamb12, TX12, C12, A12,...
%     lamb_12, TX_12, C_12, A_12,lamb13, TX13, C13, A13);
% ttt(12)=toc;
% ............
        [ V_2 ] = inter_test( TX,lamb2,TX2,C2,A2 );
        [ V3 ] = inter_test( TX,lamb3,TX3,C3,A3 );
        [ V_3 ] = inter_test( TX,lamb_3,TX_3,C_3,A_3 );
        [ V4 ] = inter_test( TX,lamb4,TX4,C4,A4 );
        [ V_4 ] = inter_test( TX,lamb_4,TX_4,C_4,A_4 );
        [ V5 ] = inter_test( TX,lamb5,TX5,C5,A5 );
        [ V_5 ] = inter_test( TX,lamb_5,TX_5,C_5,A_5 );
        [ V6 ] = inter_test( TX,lamb6,TX6,C6,A6 );
        [ V_6 ] = inter_test( TX,lamb_6,TX_6,C_6,A_6 );
        [ V7 ] = inter_test( TX,lamb7,TX7,C7,A7 );
        [ V_7 ] = inter_test( TX,lamb_7,TX_7,C_7,A_7 );
        [ V8 ] = inter_test( TX,lamb8,TX8,C8,A8 );
        [ V_8 ] = inter_test( TX,lamb_8,TX_8,C_8,A_8 );
        [ V9 ] = inter_test( TX,lamb9,TX9,C9,A9 );
        [ V_9 ] = inter_test( TX,lamb_9,TX_9,C_9,A_9 );
        [ V10 ] = inter_test( TX,lamb10,TX10,C10,A10 );
        [ V_10 ] = inter_test( TX,lamb_10,TX_10,C_10,A_10 );
        [ V11 ] = inter_test( TX,lamb11,TX11,C11,A11 );
%         [ V_11 ] = inter_test( TX,lamb_11,TX_11,C_11,A_11 );
%         [ V12 ] = inter_test( TX,lamb12,TX12,C12,A12 );
%         [ V_12 ] = inter_test( TX,lamb_12,TX_12,C_12,A_12 );
%         [ V13 ] = inter_test( TX,lamb13,TX13,C13,A13 );
%         [ V_13 ] = inter_test( TX,lamb_13,TX_13,C_13,A_13 );
%         [ V14 ] = inter_test( TX,lamb14,TX14,C14,A14 );
        
U=V3-V_2;
U1=V4-V_3;
U2=V5-V_4;
U3=V6-V_5;
U4=V7-V_6;
U5=V8-V_7;
U6=V9-V_8;
U7=V10-V_9;
U8=V11-V_10;
% U9=V12-V_11;
% U10=V13-V_12;
% U11=V14-V_13;

[ AP ] = ECP( TX, r, sigma, T, E);
% 
[m,n] = size(U);
MuSIK = zeros(m,9);
MuSIK(:,1)=U;
MuSIK(:,2)=U+U1;
MuSIK(:,3)=U+U1+U2;
MuSIK(:,4)=U+U1+U2+U3;
MuSIK(:,5)=U+U1+U2+U3+U4;
MuSIK(:,6)=U+U1+U2+U3+U4+U5;
MuSIK(:,7)=U+U1+U2+U3+U4+U5+U6;
MuSIK(:,8)=U+U1+U2+U3+U4+U5+U6+U7;
MuSIK(:,9)=U+U1+U2+U3+U4+U5+U6+U7+U8;
% MuSIK(:,10)=U+U1+U2+U3+U4+U5+U6+U7+U8+U9;
% MuSIK(:,11)=U+U1+U2+U3+U4+U5+U6+U7+U8+U9+U10;
% MuSIK(:,12)=U+U1+U2+U3+U4+U5+U6+U7+U8+U9+U10+U11;

[RMS, MAX] = deal(ones(9,1));
for i=1:9
    RMS(i)=rms( MuSIK(:,i) - AP );
    MAX(i)=max( abs( MuSIK(:,i) - AP ) );
end

