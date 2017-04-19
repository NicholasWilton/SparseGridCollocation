% clear all;
% sparse grid collocation for 2D heat eqation ( time * 1D space dimension )
% coef stands for the connection constant
coef = 2; 

inx1=[0,0]; % domain is [ inx1(1) inx2(1) ] * [ inx1(2) inx2(2) ]
inx2=[1,1]; %
d=2; % dimension

ch=200;

t=linspace(inx1(1,1),inx2(1,1),ch);
x=linspace(inx1(1,2),inx2(1,2),ch);

[X,Y]=meshgrid(t,x);
XY=[X(:) Y(:)]; % XY is the testing points
% na or nb = level n + dimension d - 1
na=3;
nb=2;

% Level 2 ....lamb stands for \lambda the coefficients, TX stands for nodes
% C stands for shape parater, A stands for scale parameter
[ lamb5, TX5, C5, A5 ] = interplant( na, d, inx1, inx2, coef );

[ lamb5_4, TX5_4, C5_4, A5_4 ] = interplant( nb, d, inx1, inx2, coef );

% Level 3
[ lamb6, TX6, C6, A6 ] = interplant( na+1, d, inx1, inx2, coef );

[ lamb6_5, TX6_5, C6_5, A6_5 ] = interplant( nb+1, d, inx1, inx2, coef );

% Level 4
[ lamb7, TX7, C7, A7 ] = interplant( na+2, d, inx1, inx2, coef );

[ lamb7_6, TX7_6, C7_6, A7_6 ] = interplant( nb+2, d, inx1, inx2, coef );

% Level 5
[ lamb8, TX8, C8, A8 ] = interplant( na+3, d, inx1, inx2, coef );

[ lamb8_7, TX8_7, C8_7, A8_7 ] = interplant( nb+3, d, inx1, inx2, coef );

% Level 6
[ lamb9, TX9, C9, A9 ] = interplant( na+4, d, inx1, inx2, coef );

[ lamb9_8, TX9_8, C9_8, A9_8 ] = interplant( nb+4, d, inx1, inx2, coef );

% Level 7
[ lamb10, TX10, C10, A10 ] = interplant( na+5, d, inx1, inx2, coef );

[ lamb10_9, TX10_9, C10_9, A10_9 ] = interplant( nb+5, d, inx1, inx2, coef );

% Level 8
[ lamb11, TX11, C11, A11 ] = interplant( na+6, d, inx1, inx2, coef );

[ lamb11_10, TX11_10, C11_10, A11_10 ] = interplant( nb+6, d, inx1, inx2, coef );

% Level 9
[ lamb12, TX12, C12, A12 ] = interplant( na+7, d, inx1, inx2, coef );

[ lamb12_11, TX12_11, C12_11, A12_11 ] = interplant( nb+7, d, inx1, inx2, coef );

% Level 10
[ lamb13, TX13, C13, A13 ] = interplant( na+8, d, inx1, inx2, coef );

[ lamb13_12, TX13_12, C13_12, A13_12 ] = interplant( nb+8, d, inx1, inx2, coef );

%.......
[ V5 ] = inter_test( XY,lamb5,TX5,C5,A5 );
[ V5_4 ] = inter_test( XY,lamb5_4,TX5_4,C5_4,A5_4 );
%.......
[ V6 ] = inter_test( XY,lamb6,TX6,C6,A6 );
[ V6_5 ] = inter_test( XY,lamb6_5,TX6_5,C6_5,A6_5 );
%.......
[ V7 ] = inter_test( XY,lamb7,TX7,C7,A7 );
[ V7_6 ] = inter_test( XY,lamb7_6,TX7_6,C7_6,A7_6 );
%.......
[ V8 ] = inter_test( XY,lamb8,TX8,C8,A8 );
[ V8_7 ] = inter_test( XY,lamb8_7,TX8_7,C8_7,A8_7 );
%.......
[ V9 ] = inter_test( XY,lamb9,TX9,C9,A9 );
[ V9_8 ] = inter_test( XY,lamb9_8,TX9_8,C9_8,A9_8 );
%.......
[ V10 ] = inter_test( XY,lamb10,TX10,C10,A10 );
[ V10_9 ] = inter_test( XY,lamb10_9,TX10_9,C10_9,A10_9 );
%.......
[ V11 ] = inter_test( XY,lamb11,TX11,C11,A11 );
[ V11_10 ] = inter_test( XY,lamb11_10,TX11_10,C11_10,A11_10 );
% %.......
[ V12 ] = inter_test( XY,lamb12,TX12,C12,A12 );
[ V12_11 ] = inter_test( XY,lamb12_11,TX12_11,C12_11,A12_11 );
% %.......
[ V13 ] = inter_test( XY,lamb13,TX13,C13,A13 );
[ V13_12 ] = inter_test( XY,lamb13_12,TX13_12,C13_12,A13_12 );
% %.......

% SIK is sparse grid collocation, depending on the combination tech
SIK(:,1)=V5-V5_4;
SIK(:,2)=V6-V6_5;
SIK(:,3)=V7-V7_6;
SIK(:,4)=V8-V8_7;
SIK(:,5)=V9-V9_8;
SIK(:,6)=V10-V10_9;
SIK(:,7)=V11-V11_10;
SIK(:,8)=V12-V12_11;
SIK(:,9)=V13-V13_12;
% f is target function
f=@(x) ( exp(-pi^2 .* x(:,1)) .* sin(pi .* x(:,2)) );
[ AP ] = f(XY);

[RMS_s, MAX_s] = deal(zeros(1,9));

for i = 1 : 9
    RMS_s(i) = rms( SIK(:,i) - AP );
    MAX_s(i) = max( abs( SIK(:,i) - AP ) ) ;
end