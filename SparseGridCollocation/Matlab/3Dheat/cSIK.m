% clear all;
% sparse grid collocation for 3D heat eqation ( time * 2D space dimensions )
% coef stands for the connection constant
coef=2;

inx1=[0,0,0]; % domain is [ inx1(1) inx2(1) ] * [ inx1(2) inx2(2) ] * [ inx1(3) inx2(3) ]
inx2=[1,1,1];
d=3; % dimension

ch=50;

t=linspace(inx1(1,1),inx2(1,1),ch);
x=linspace(inx1(1,2),inx2(1,2),ch);
y=linspace(inx1(1,3),inx2(1,3),ch);

[X,Y,Z]=meshgrid(t,x,y);
TXY=[X(:) Y(:) Z(:)]; % TXY is the testing points
% na or nb or nc = level n + dimension d - 1
na=5;
nb=4;
nc=3;

% Level 3 ....lamb stands for \lambda the coefficients, TX stands for nodes
% C stands for shape parater, A stands for scale parameter
[ lamb5, TX5, C5, A5 ] = interplant( na, d, inx1, inx2, coef );

[ lamb5_4, TX5_4, C5_4, A5_4 ] = interplant( nb, d, inx1, inx2, coef );

[ lamb5_3, TX5_3, C5_3, A5_3 ] = interplant( nc, d, inx1, inx2, coef );
% Level 4
[ lamb6, TX6, C6, A6 ] = interplant( na+1, d, inx1, inx2, coef );

[ lamb6_5, TX6_5, C6_5, A6_5 ] = interplant( nb+1, d, inx1, inx2, coef );

[ lamb6_4, TX6_4, C6_4, A6_4 ] = interplant( nc+1, d, inx1, inx2, coef );
% Level 5
[ lamb7, TX7, C7, A7 ] = interplant( na+2, d, inx1, inx2, coef );

[ lamb7_6, TX7_6, C7_6, A7_6 ] = interplant( nb+2, d, inx1, inx2, coef );

[ lamb7_5, TX7_5, C7_5, A7_5 ] = interplant( nc+2, d, inx1, inx2, coef );
% Level 6
[ lamb8, TX8, C8, A8 ] = interplant( na+3, d, inx1, inx2, coef );

[ lamb8_7, TX8_7, C8_7, A8_7 ] = interplant( nb+3, d, inx1, inx2, coef );

[ lamb8_6, TX8_6, C8_6, A8_6 ] = interplant( nc+3, d, inx1, inx2, coef );
% Level 7
[ lamb9, TX9, C9, A9 ] = interplant( na+4, d, inx1, inx2, coef );

[ lamb9_8, TX9_8, C9_8, A9_8 ] = interplant( nb+4, d, inx1, inx2, coef );

[ lamb9_7, TX9_7, C9_7, A9_7 ] = interplant( nc+4, d, inx1, inx2, coef );
% Level 8
[ lamb10, TX10, C10, A10 ] = interplant( na+5, d, inx1, inx2, coef );

[ lamb10_9, TX10_9, C10_9, A10_9 ] = interplant( nb+5, d, inx1, inx2, coef );

[ lamb10_8, TX10_8, C10_8, A10_8 ] = interplant( nc+5, d, inx1, inx2, coef );
% Level 9
% [ lamb11, TX11, C11, A11 ] = interplant( na+6, d, inx1, inx2, coef );
% 
% [ lamb11_10, TX11_10, C11_10, A11_10 ] = interplant( nb+6, d, inx1, inx2, coef );
% 
% [ lamb11_9, TX11_9, C11_9, A11_9 ] = interplant( nc+6, d, inx1, inx2, coef );
% Level 10
% [ lamb12, TX12, C12, A12 ] = interplant( na+7, d, inx1, inx2, coef );
% 
% [ lamb12_11, TX12_11, C12_11, A12_11 ] = interplant( nb+7, d, inx1, inx2, coef );
% 
% [ lamb12_10, TX12_10, C12_10, A12_10 ] = interplant( nc+7, d, inx1, inx2, coef );
%.......

%.......
[ V5 ] = inter_test( TXY,lamb5,TX5,C5,A5 );
[ V5_4 ] = inter_test( TXY,lamb5_4,TX5_4,C5_4,A5_4 );
[ V5_3 ] = inter_test( TXY,lamb5_3,TX5_3,C5_3,A5_3 );
%.......
[ V6 ] = inter_test( TXY,lamb6,TX6,C6,A6 );
[ V6_5 ] = inter_test( TXY,lamb6_5,TX6_5,C6_5,A6_5 );
[ V6_4 ] = inter_test( TXY,lamb6_4,TX6_4,C6_4,A6_4 );
%.......
[ V7 ] = inter_test( TXY,lamb7,TX7,C7,A7 );
[ V7_6 ] = inter_test( TXY,lamb7_6,TX7_6,C7_6,A7_6 );
[ V7_5 ] = inter_test( TXY,lamb7_5,TX7_5,C7_5,A7_5 );
%.......
[ V8 ] = inter_test( TXY,lamb8,TX8,C8,A8 );
[ V8_7 ] = inter_test( TXY,lamb8_7,TX8_7,C8_7,A8_7 );
[ V8_6 ] = inter_test( TXY,lamb8_6,TX8_6,C8_6,A8_6 );
%.......
[ V9 ] = inter_test( TXY,lamb9,TX9,C9,A9 );
[ V9_8 ] = inter_test( TXY,lamb9_8,TX9_8,C9_8,A9_8 );
[ V9_7 ] = inter_test( TXY,lamb9_7,TX9_7,C9_7,A9_7 );
%.......
[ V10 ] = inter_test( TXY,lamb10,TX10,C10,A10 );
[ V10_9 ] = inter_test( TXY,lamb10_9,TX10_9,C10_9,A10_9 );
[ V10_8 ] = inter_test( TXY,lamb10_8,TX10_8,C10_8,A10_8 );
%.......
% [ V11 ] = inter_test( TXY,lamb11,TX11,C11,A11 );
% [ V11_10 ] = inter_test( TXY,lamb11_10,TX11_10,C11_10,A11_10 );
% [ V11_9 ] = inter_test( TXY,lamb11_9,TX11_9,C11_9,A11_9 );
% %.......
% [ V12 ] = inter_test( TXY,lamb12,TX12,C12,A12 );
% [ V12_11 ] = inter_test( TXY,lamb12_11,TX12_11,C12_11,A12_11 );
% [ V12_10 ] = inter_test( TXY,lamb12_10,TX12_10,C12_10,A12_10 );
% %.......


SIK(:,1) = V5-2*V5_4+V5_3;
SIK(:,2) = V6-2*V6_5+V6_4;
SIK(:,3) = V7-2*V7_6+V7_5;
SIK(:,4) = V8-2*V8_7+V8_6;
SIK(:,5) = V9-2*V9_8+V9_7;
SIK(:,6) = V10-2*V10_9+V10_8;
% SIK(:,7) = V11-2*V11_10+V11_9;
% SIK(:,8) = V12-2*V12_11+V12_10;

% f =@(X) (exp(-2*pi^2 .* X(:,1)) .* sin(pi .* X(:,2)) .* cos(pi .* X(:,3)));
% [ AP ] = f(TXY);

f =@(x) sin(pi*x(:,1)).*sin(pi*x(:,2)).*sin(pi*x(:,3));
[ AP ] = f(TXY);
[RMS_s, MAX_s] = deal(zeros(1,6));
for i = 1 : 6
    RMS_s(i) = rms( SIK(:,i) - AP );
    MAX_s(i) = max( abs( SIK(:,i) - AP ) );
end