% sparse grid collocation for 2D BS ( time * 1D stock dimension )
tic
parpool local
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
% Level 2 ....lamb stands for \lambda the coefficients, TX stands for nodes
% C stands for shape parater, A stands for scale parameter
[ lamb3, TX3, C3, A3 ] = interplant( coef, tsec, na, d, inx1, inx2, r, sigma, T, E );

[ lamb2, TX2, C2, A2 ] = interplant( coef, tsec, nb, d, inx1, inx2, r, sigma, T, E );
ttt(1) = toc;
% .........
% Level 3
[ lamb4, TX4, C4, A4 ] = interplant( coef, tsec, na+1, d, inx1, inx2, r, sigma, T, E );

[ lamb_3, TX_3, C_3, A_3 ] = interplant( coef, tsec, nb+1, d, inx1, inx2, r, sigma, T, E );
ttt(2) = toc;
% .........
% Level 4
[ lamb5, TX5, C5, A5 ] = interplant( coef, tsec, na+2, d, inx1, inx2, r, sigma, T, E );

[ lamb_4, TX_4, C_4, A_4 ] = interplant( coef, tsec, nb+2, d, inx1, inx2, r, sigma, T, E );
ttt(3) = toc;
% .........
% Level 5
[ lamb6, TX6, C6, A6 ] = interplant( coef, tsec, na+3, d, inx1, inx2, r, sigma, T, E );

[ lamb_5, TX_5, C_5, A_5 ] = interplant( coef, tsec, nb+3, d, inx1, inx2, r, sigma, T, E );
ttt(4) = toc;
% .........
% Level 6
[ lamb7, TX7, C7, A7 ] = interplant( coef, tsec, na+4, d, inx1, inx2, r, sigma, T, E );

[ lamb_6, TX_6, C_6, A_6 ] = interplant( coef, tsec, nb+4, d, inx1, inx2, r, sigma, T, E );
ttt(5) = toc;
% .........
% Level 7
[ lamb8, TX8, C8, A8 ] = interplant( coef, tsec, na+5, d, inx1, inx2, r, sigma, T, E );

[ lamb_7, TX_7, C_7, A_7 ] = interplant( coef, tsec, nb+5, d, inx1, inx2, r, sigma, T, E );
ttt(6) = toc;
% .........
% Level 8
[ lamb9, TX9, C9, A9 ] = interplant( coef, tsec, na+6, d, inx1, inx2, r, sigma, T, E );

[ lamb_8, TX_8, C_8, A_8 ] = interplant( coef, tsec, nb+6, d, inx1, inx2, r, sigma, T, E );
ttt(7) = toc;
% .........
% Level 9
[ lamb10, TX10, C10, A10 ] = interplant( coef, tsec, na+7, d, inx1, inx2, r, sigma, T, E );

[ lamb_9, TX_9, C_9, A_9 ] = interplant( coef, tsec, nb+7, d, inx1, inx2, r, sigma, T, E );
ttt(8) = toc;
% .........
% Level 10
[ lamb11, TX11, C11, A11 ] = interplant( coef, tsec, na+8, d, inx1, inx2, r, sigma, T, E );

[ lamb_10, TX_10, C_10, A_10 ] = interplant( coef, tsec, nb+8, d, inx1, inx2, r, sigma, T, E );
ttt(9) = toc;
% .........
% Level 11
% [ lamb12, TX12, C12, A12 ] = interplant( coef, tsec, na+9, d, inx1, inx2, r, sigma, T, E );
% 
% [ lamb_11, TX_11, C_11, A_11 ] = interplant( coef, tsec, nb+9, d, inx1, inx2, r, sigma, T, E );
% .........
% Level 12
% [ lamb13, TX13, C13, A13 ] = interplant( coef, tsec, na+10, d, inx1, inx2, r, sigma, T, E );
% 
% [ lamb_12, TX_12, C_12, A_12 ] = interplant( coef, tsec, nb+10, d, inx1, inx2, r, sigma, T, E );
% .........
% Level 13
% [ lamb14, TX14, C14, A14 ] = interplant( coef, tsec, na+11, d, inx1, inx2, r, sigma, T, E );
% 
% [ lamb_13, TX_13, C_13, A_13 ] = interplant( coef, tsec, nb+11, d, inx1, inx2, r, sigma, T, E );

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
    
[m,n] = size(V3);
SIK = zeros(m,9);
SIK(:,1)=V3-V_2;
SIK(:,2)=V4-V_3;
SIK(:,3)=V5-V_4;
SIK(:,4)=V6-V_5;
SIK(:,5)=V7-V_6;
SIK(:,6)=V8-V_7;
SIK(:,7)=V9-V_8;
SIK(:,8)=V10-V_9;
SIK(:,9)=V11-V_10;
% SIK(:,10)=V12-V_11;
% SIK(:,11)=V13-V_12;
% SIK(:,12)=V14-V_13;
delete(gcp('nocreate'))
ttt(10)=toc;
[ AP ] = ECP( TX, r, sigma, T, E);
[RMS_s, MAX_s] = deal(ones(9,1));
for i=1:9
    RMS_s(i)=rms( SIK(:,i) - AP );
    MAX_s(i)=max( abs( SIK(:,i) - AP ) );
end

[muRows, muCols] = size(SIK);
[rRows, rCols] = size(RMS);
[mRows, mCols] = size(MAX);
[tRows, tCols] = size(ttt);
saveMatrixB("MatLab_SIKc_Results_" + muRows + "_" + muCols + ".dat",SIK);
saveMatrixB("MatLab_SIKc_RMS_" + rRows + "_" + rCols + ".dat",RMS);
saveMatrixB("MatLab_SIKc_MAX_" + mRows + "_" + mCols + ".dat",MAX);
saveMatrixB("MatLab_SIKc_Timings_" + tRows + "_" + tCols + ".dat",ttt);

