% clear all;
% sparse grid collocation for 4D heat eqation ( time * 3D space dimensions )
% coef stands for the connection constant
coef=2;

inx1=[0,0,0,0]; % domain is [ inx1(1) inx2(1) ] * [ inx1(2) inx2(2) ] 
inx2=[1,1,1,1]; %           * [ inx1(3) inx2(3) ] * [ inx1(4) inx2(4) ]
d=4; % dimension

ch=22;

t=linspace(inx1(1,1),inx2(1,1),ch);
x=linspace(inx1(1,2),inx2(1,2),ch);
y=linspace(inx1(1,3),inx2(1,3),ch);
z=linspace(inx1(1,4),inx2(1,4),ch);

[V1, V2, V3, V4] = ndgrid(t,x,y,z);
V1 = reshape(V1,[],1);
V2 = reshape(V2,[],1);
V3 = reshape(V3,[],1);
V4 = reshape(V4,[],1);
TXYZ = [V1 V2 V3 V4]; % TXYZ is the testing points
% na, nb, nc, nd = level n + dimension d - 1
na=7;
nb=6;
nc=5;
nd=4;

% Level 4 ....lamb stands for \lambda the coefficients, TX stands for nodes
% C stands for shape parater, A stands for scale parameter
[ lamb7, TX7, C7, A7 ] = interplant( na, d, inx1, inx2, coef );

[ lamb7_6, TX7_6, C7_6, A7_6 ] = interplant( nb, d, inx1, inx2, coef );

[ lamb7_5, TX7_5, C7_5, A7_5 ] = interplant( nc, d, inx1, inx2, coef );

[ lamb7_4, TX7_4, C7_4, A7_4 ] = interplant( nd, d, inx1, inx2, coef );

% Level 5
[ lamb8, TX8, C8, A8 ] = interplant( na+1, d, inx1, inx2, coef );

[ lamb8_7, TX8_7, C8_7, A8_7 ] = interplant( nb+1, d, inx1, inx2, coef );

[ lamb8_6, TX8_6, C8_6, A8_6 ] = interplant( nc+1, d, inx1, inx2, coef );

[ lamb8_5, TX8_5, C8_5, A8_5 ] = interplant( nd+1, d, inx1, inx2, coef );
% Level 6
[ lamb9, TX9, C9, A9 ] = interplant( na+2, d, inx1, inx2, coef );

[ lamb9_8, TX9_8, C9_8, A9_8 ] = interplant( nb+2, d, inx1, inx2, coef );

[ lamb9_7, TX9_7, C9_7, A9_7 ] = interplant( nc+2, d, inx1, inx2, coef );

[ lamb9_6, TX9_6, C9_6, A9_6 ] = interplant( nd+2, d, inx1, inx2, coef );
% Level 7
[ lamb10, TX10, C10, A10 ] = interplant( na+3, d, inx1, inx2, coef );

[ lamb10_9, TX10_9, C10_9, A10_9 ] = interplant( nb+3, d, inx1, inx2, coef );

[ lamb10_8, TX10_8, C10_8, A10_8 ] = interplant( nc+3, d, inx1, inx2, coef );

[ lamb10_7, TX10_7, C10_7, A10_7 ] = interplant( nd+3, d, inx1, inx2, coef );
% Level 8
[ lamb11, TX11, C11, A11 ] = interplant( na+4, d, inx1, inx2, coef );

[ lamb11_10, TX11_10, C11_10, A11_10 ] = interplant( nb+4, d, inx1, inx2, coef );

[ lamb11_9, TX11_9, C11_9, A11_9 ] = interplant( nc+4, d, inx1, inx2, coef );

[ lamb11_8, TX11_8, C11_8, A11_8 ] = interplant( nd+4, d, inx1, inx2, coef );
% Level 9
% [ lamb12, TX12, C12, A12 ] = interplant( na+5, d, inx1, inx2, coef );
% 
% [ lamb12_11, TX12_11, C12_11, A12_11 ] = interplant( nb+5, d, inx1, inx2, coef );
% 
% [ lamb12_10, TX12_10, C12_10, A12_10 ] = interplant( nc+5, d, inx1, inx2, coef );
% 
% [ lamb12_9, TX12_9, C12_9, A12_9 ] = interplant( nd+5, d, inx1, inx2, coef );
% .......

%.......
[ V7 ] = inter_test( TXYZ,lamb7,TX7,C7,A7 );
[ V7_6 ] = inter_test( TXYZ,lamb7_6,TX7_6,C7_6,A7_6 );
[ V7_5 ] = inter_test( TXYZ,lamb7_5,TX7_5,C7_5,A7_5 );
[ V7_4 ] = inter_test( TXYZ,lamb7_4,TX7_4,C7_4,A7_4 );
%.......
[ V8 ] = inter_test( TXYZ,lamb8,TX8,C8,A8 );
[ V8_7 ] = inter_test( TXYZ,lamb8_7,TX8_7,C8_7,A8_7 );
[ V8_6 ] = inter_test( TXYZ,lamb8_6,TX8_6,C8_6,A8_6 );
[ V8_5 ] = inter_test( TXYZ,lamb8_5,TX8_5,C8_5,A8_5 );
%.......
[ V9 ] = inter_test( TXYZ,lamb9,TX9,C9,A9 );
[ V9_8 ] = inter_test( TXYZ,lamb9_8,TX9_8,C9_8,A9_8 );
[ V9_7 ] = inter_test( TXYZ,lamb9_7,TX9_7,C9_7,A9_7 );
[ V9_6 ] = inter_test( TXYZ,lamb9_6,TX9_6,C9_6,A9_6 );
%.......
[ V10 ] = inter_test( TXYZ,lamb10,TX10,C10,A10 );
[ V10_9 ] = inter_test( TXYZ,lamb10_9,TX10_9,C10_9,A10_9 );
[ V10_8 ] = inter_test( TXYZ,lamb10_8,TX10_8,C10_8,A10_8 );
[ V10_7 ] = inter_test( TXYZ,lamb10_7,TX10_7,C10_7,A10_7 );
%.......
[ V11 ] = inter_test( TXYZ,lamb11,TX11,C11,A11 );
[ V11_10 ] = inter_test( TXYZ,lamb11_10,TX11_10,C11_10,A11_10 );
[ V11_9 ] = inter_test( TXYZ,lamb11_9,TX11_9,C11_9,A11_9 );
[ V11_8 ] = inter_test( TXYZ,lamb11_8,TX11_8,C11_8,A11_8 );
%.......
% [ V12 ] = inter_test( TXYZ,lamb12,TX12,C12,A12 );
% [ V12_11 ] = inter_test( TXYZ,lamb12_11,TX12_11,C12_11,A12_11 );
% [ V12_10 ] = inter_test( TXYZ,lamb12_10,TX12_10,C12_10,A12_10 );
% [ V12_9 ] = inter_test( TXYZ,lamb12_9,TX12_9,C12_9,A12_9 );
%.......

SIK(:,1) = V7-3*V7_6+3*V7_5-V7_4;
SIK(:,2) = V8-3*V8_7+3*V8_6-V8_5;
SIK(:,3) = V9-3*V9_8+3*V9_7-V9_6;
SIK(:,4) = V10-3*V10_9+3*V10_8-V10_7;
SIK(:,5) = V11-3*V11_10+3*V11_9-V11_8;
% SIK(:,6) = V12-3*V12_11+3*V12_10-V12_9;

f = @(x) sin(pi*x(:,1)) .* sin(pi*x(:,2)) .* sin(pi*x(:,3)) .* sin(pi*x(:,4));
[ AP ] = f(TXYZ);
[RMS_s, MAX_s] = deal(zeros(1,5));
for i = 1 : 5
    RMS_s(i) = rms( SIK(:,i) - AP );
    MAX_s(i) = max( abs( SIK(:,i) - AP ) );
end
