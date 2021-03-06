% clear all;
% multilevel sparse grid collocation for 3D heat eqation ( time * 2D space dimensions )
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
% Level 4 .... multilevel method has to use all previous information
[ lamb6, TX6, C6, A6 ] = interplant_1( na+1, d, inx1, inx2, coef, lamb5, TX5, C5, A5,...
    lamb5_4, TX5_4, C5_4, A5_4, lamb5_3, TX5_3, C5_3, A5_3);

[ lamb6_5, TX6_5, C6_5, A6_5 ] = interplant_1( nb+1, d, inx1, inx2, coef, lamb5, TX5, C5, A5,...
    lamb5_4, TX5_4, C5_4, A5_4, lamb5_3, TX5_3, C5_3, A5_3);

[ lamb6_4, TX6_4, C6_4, A6_4 ] = interplant_1( nc+1, d, inx1, inx2, coef, lamb5, TX5, C5, A5,...
    lamb5_4, TX5_4, C5_4, A5_4, lamb5_3, TX5_3, C5_3, A5_3 );
% Level 5 .... higher level needs more information
[ lamb7, TX7, C7, A7 ] = interplant_2( na+2, d, inx1, inx2, coef, lamb5, TX5, C5, A5,...
    lamb5_4, TX5_4, C5_4, A5_4, lamb5_3, TX5_3, C5_3, A5_3, lamb6, TX6, C6, A6,...
    lamb6_5, TX6_5, C6_5, A6_5, lamb6_4, TX6_4, C6_4, A6_4);

[ lamb7_6, TX7_6, C7_6, A7_6 ] = interplant_2( nb+2, d, inx1, inx2, coef, lamb5, TX5, C5, A5,...
    lamb5_4, TX5_4, C5_4, A5_4, lamb5_3, TX5_3, C5_3, A5_3,lamb6, TX6, C6, A6,...
    lamb6_5, TX6_5, C6_5, A6_5, lamb6_4, TX6_4, C6_4, A6_4 );

[ lamb7_5, TX7_5, C7_5, A7_5 ] = interplant_2( nc+2, d, inx1, inx2, coef, lamb5, TX5, C5, A5,...
    lamb5_4, TX5_4, C5_4, A5_4, lamb5_3, TX5_3, C5_3, A5_3,lamb6, TX6, C6, A6,...
    lamb6_5, TX6_5, C6_5, A6_5, lamb6_4, TX6_4, C6_4, A6_4 );
% Level 6
[ lamb8, TX8, C8, A8 ] = interplant_3( na+3, d, inx1, inx2, coef, lamb5, TX5, C5, A5,...
    lamb5_4, TX5_4, C5_4, A5_4, lamb5_3, TX5_3, C5_3, A5_3,lamb6, TX6, C6, A6,...
    lamb6_5, TX6_5, C6_5, A6_5, lamb6_4, TX6_4, C6_4, A6_4, lamb7, TX7, C7, A7,...
    lamb7_6, TX7_6, C7_6, A7_6, lamb7_5, TX7_5, C7_5, A7_5 );

[ lamb8_7, TX8_7, C8_7, A8_7 ] = interplant_3( nb+3, d, inx1, inx2, coef, lamb5, TX5, C5, A5,...
    lamb5_4, TX5_4, C5_4, A5_4, lamb5_3, TX5_3, C5_3, A5_3,lamb6, TX6, C6, A6,...
    lamb6_5, TX6_5, C6_5, A6_5, lamb6_4, TX6_4, C6_4, A6_4, lamb7, TX7, C7, A7,...
    lamb7_6, TX7_6, C7_6, A7_6, lamb7_5, TX7_5, C7_5, A7_5 );

[ lamb8_6, TX8_6, C8_6, A8_6 ] = interplant_3( nc+3, d, inx1, inx2, coef, lamb5, TX5, C5, A5,...
    lamb5_4, TX5_4, C5_4, A5_4, lamb5_3, TX5_3, C5_3, A5_3,lamb6, TX6, C6, A6,...
    lamb6_5, TX6_5, C6_5, A6_5, lamb6_4, TX6_4, C6_4, A6_4, lamb7, TX7, C7, A7,...
    lamb7_6, TX7_6, C7_6, A7_6, lamb7_5, TX7_5, C7_5, A7_5 );
% Level 7
[ lamb9, TX9, C9, A9 ] = interplant_4( na+4, d, inx1, inx2, coef, lamb5, TX5, C5, A5,...
    lamb5_4, TX5_4, C5_4, A5_4, lamb5_3, TX5_3, C5_3, A5_3,lamb6, TX6, C6, A6,...
    lamb6_5, TX6_5, C6_5, A6_5, lamb6_4, TX6_4, C6_4, A6_4, lamb7, TX7, C7, A7,...
    lamb7_6, TX7_6, C7_6, A7_6, lamb7_5, TX7_5, C7_5, A7_5, lamb8, TX8, C8, A8,...
    lamb8_7, TX8_7, C8_7, A8_7, lamb8_6, TX8_6, C8_6, A8_6);

[ lamb9_8, TX9_8, C9_8, A9_8 ] = interplant_4( nb+4, d, inx1, inx2, coef, lamb5, TX5, C5, A5,...
    lamb5_4, TX5_4, C5_4, A5_4, lamb5_3, TX5_3, C5_3, A5_3,lamb6, TX6, C6, A6,...
    lamb6_5, TX6_5, C6_5, A6_5, lamb6_4, TX6_4, C6_4, A6_4, lamb7, TX7, C7, A7,...
    lamb7_6, TX7_6, C7_6, A7_6, lamb7_5, TX7_5, C7_5, A7_5, lamb8, TX8, C8, A8,...
    lamb8_7, TX8_7, C8_7, A8_7, lamb8_6, TX8_6, C8_6, A8_6 );

[ lamb9_7, TX9_7, C9_7, A9_7 ] = interplant_4( nc+4, d, inx1, inx2, coef, lamb5, TX5, C5, A5,...
    lamb5_4, TX5_4, C5_4, A5_4, lamb5_3, TX5_3, C5_3, A5_3,lamb6, TX6, C6, A6,...
    lamb6_5, TX6_5, C6_5, A6_5, lamb6_4, TX6_4, C6_4, A6_4, lamb7, TX7, C7, A7,...
    lamb7_6, TX7_6, C7_6, A7_6, lamb7_5, TX7_5, C7_5, A7_5, lamb8, TX8, C8, A8,...
    lamb8_7, TX8_7, C8_7, A8_7, lamb8_6, TX8_6, C8_6, A8_6 );
% Level 8
[ lamb10, TX10, C10, A10 ] = interplant_5( na+5, d, inx1, inx2, coef, lamb5, TX5, C5, A5,...
    lamb5_4, TX5_4, C5_4, A5_4, lamb5_3, TX5_3, C5_3, A5_3,lamb6, TX6, C6, A6,...
    lamb6_5, TX6_5, C6_5, A6_5, lamb6_4, TX6_4, C6_4, A6_4, lamb7, TX7, C7, A7,...
    lamb7_6, TX7_6, C7_6, A7_6, lamb7_5, TX7_5, C7_5, A7_5, lamb8, TX8, C8, A8,...
    lamb8_7, TX8_7, C8_7, A8_7, lamb8_6, TX8_6, C8_6, A8_6, lamb9, TX9, C9, A9,...
    lamb9_8, TX9_8, C9_8, A9_8, lamb9_7, TX9_7, C9_7, A9_7 );

[ lamb10_9, TX10_9, C10_9, A10_9 ] = interplant_5( nb+5, d, inx1, inx2, coef, lamb5, TX5, C5, A5,...
    lamb5_4, TX5_4, C5_4, A5_4, lamb5_3, TX5_3, C5_3, A5_3,lamb6, TX6, C6, A6,...
    lamb6_5, TX6_5, C6_5, A6_5, lamb6_4, TX6_4, C6_4, A6_4, lamb7, TX7, C7, A7,...
    lamb7_6, TX7_6, C7_6, A7_6, lamb7_5, TX7_5, C7_5, A7_5,lamb8, TX8, C8, A8,...
    lamb8_7, TX8_7, C8_7, A8_7, lamb8_6, TX8_6, C8_6, A8_6, lamb9, TX9, C9, A9,...
    lamb9_8, TX9_8, C9_8, A9_8, lamb9_7, TX9_7, C9_7, A9_7 );

[ lamb10_8, TX10_8, C10_8, A10_8 ] = interplant_5( nc+5, d, inx1, inx2, coef, lamb5, TX5, C5, A5,...
    lamb5_4, TX5_4, C5_4, A5_4, lamb5_3, TX5_3, C5_3, A5_3,lamb6, TX6, C6, A6,...
    lamb6_5, TX6_5, C6_5, A6_5, lamb6_4, TX6_4, C6_4, A6_4, lamb7, TX7, C7, A7,...
    lamb7_6, TX7_6, C7_6, A7_6, lamb7_5, TX7_5, C7_5, A7_5, lamb8, TX8, C8, A8,...
    lamb8_7, TX8_7, C8_7, A8_7, lamb8_6, TX8_6, C8_6, A8_6, lamb9, TX9, C9, A9,...
    lamb9_8, TX9_8, C9_8, A9_8, lamb9_7, TX9_7, C9_7, A9_7 );
% Level 9
% [ lamb11, TX11, C11, A11 ] = interplant_6( na+6, d, inx1, inx2, coef, lamb5, TX5, C5, A5,...
%     lamb5_4, TX5_4, C5_4, A5_4, lamb5_3, TX5_3, C5_3, A5_3,lamb6, TX6, C6, A6,...
%     lamb6_5, TX6_5, C6_5, A6_5, lamb6_4, TX6_4, C6_4, A6_4, lamb7, TX7, C7, A7,...
%     lamb7_6, TX7_6, C7_6, A7_6, lamb7_5, TX7_5, C7_5, A7_5, lamb8, TX8, C8, A8,...
%     lamb8_7, TX8_7, C8_7, A8_7, lamb8_6, TX8_6, C8_6, A8_6, lamb9, TX9, C9, A9,...
%     lamb9_8, TX9_8, C9_8, A9_8, lamb9_7, TX9_7, C9_7, A9_7, lamb10, TX10, C10, A10,...
%     lamb10_9, TX10_9, C10_9, A10_9, lamb10_8, TX10_8, C10_8, A10_8 );
% 
% [ lamb11_10, TX11_10, C11_10, A11_10 ] = interplant_6( nb+6, d, inx1, inx2, coef, lamb5, TX5, C5, A5,...
%     lamb5_4, TX5_4, C5_4, A5_4, lamb5_3, TX5_3, C5_3, A5_3,lamb6, TX6, C6, A6,...
%     lamb6_5, TX6_5, C6_5, A6_5, lamb6_4, TX6_4, C6_4, A6_4, lamb7, TX7, C7, A7,...
%     lamb7_6, TX7_6, C7_6, A7_6, lamb7_5, TX7_5, C7_5, A7_5, lamb8, TX8, C8, A8,...
%     lamb8_7, TX8_7, C8_7, A8_7, lamb8_6, TX8_6, C8_6, A8_6, lamb9, TX9, C9, A9,...
%     lamb9_8, TX9_8, C9_8, A9_8, lamb9_7, TX9_7, C9_7, A9_7, lamb10, TX10, C10, A10,...
%     lamb10_9, TX10_9, C10_9, A10_9, lamb10_8, TX10_8, C10_8, A10_8 );
% 
% [ lamb11_9, TX11_9, C11_9, A11_9 ] = interplant_6( nc+6, d, inx1, inx2, coef, lamb5, TX5, C5, A5,...
%     lamb5_4, TX5_4, C5_4, A5_4, lamb5_3, TX5_3, C5_3, A5_3,lamb6, TX6, C6, A6,...
%     lamb6_5, TX6_5, C6_5, A6_5, lamb6_4, TX6_4, C6_4, A6_4, lamb7, TX7, C7, A7,...
%     lamb7_6, TX7_6, C7_6, A7_6, lamb7_5, TX7_5, C7_5, A7_5, lamb8, TX8, C8, A8,...
%     lamb8_7, TX8_7, C8_7, A8_7, lamb8_6, TX8_6, C8_6, A8_6, lamb9, TX9, C9, A9,...
%     lamb9_8, TX9_8, C9_8, A9_8, lamb9_7, TX9_7, C9_7, A9_7, lamb10, TX10, C10, A10,...
%     lamb10_9, TX10_9, C10_9, A10_9, lamb10_8, TX10_8, C10_8, A10_8 );
% Level 10
% [ lamb12, TX12, C12, A12 ] = interplant_7( na+7, d, inx1, inx2, coef, lamb5, TX5, C5, A5,...
%     lamb5_4, TX5_4, C5_4, A5_4, lamb5_3, TX5_3, C5_3, A5_3,lamb6, TX6, C6, A6,...
%     lamb6_5, TX6_5, C6_5, A6_5, lamb6_4, TX6_4, C6_4, A6_4, lamb7, TX7, C7, A7,...
%     lamb7_6, TX7_6, C7_6, A7_6, lamb7_5, TX7_5, C7_5, A7_5, lamb8, TX8, C8, A8,...
%     lamb8_7, TX8_7, C8_7, A8_7, lamb8_6, TX8_6, C8_6, A8_6, lamb9, TX9, C9, A9,...
%     lamb9_8, TX9_8, C9_8, A9_8, lamb9_7, TX9_7, C9_7, A9_7, lamb10, TX10, C10, A10,...
%     lamb10_9, TX10_9, C10_9, A10_9, lamb10_8, TX10_8, C10_8, A10_8, lamb11, TX11, C11, A11,...
%     lamb11_10, TX11_10, C11_10, A11_10, lamb11_9, TX11_9, C11_9, A11_9 );
% 
% [ lamb12_11, TX12_11, C12_11, A12_11 ] = interplant_7( nb+7, d, inx1, inx2, coef, lamb5, TX5, C5, A5,...
%     lamb5_4, TX5_4, C5_4, A5_4, lamb5_3, TX5_3, C5_3, A5_3,lamb6, TX6, C6, A6,...
%     lamb6_5, TX6_5, C6_5, A6_5, lamb6_4, TX6_4, C6_4, A6_4, lamb7, TX7, C7, A7,...
%     lamb7_6, TX7_6, C7_6, A7_6, lamb7_5, TX7_5, C7_5, A7_5, lamb8, TX8, C8, A8,...
%     lamb8_7, TX8_7, C8_7, A8_7, lamb8_6, TX8_6, C8_6, A8_6, lamb9, TX9, C9, A9,...
%     lamb9_8, TX9_8, C9_8, A9_8, lamb9_7, TX9_7, C9_7, A9_7, lamb10, TX10, C10, A10,...
%     lamb10_9, TX10_9, C10_9, A10_9, lamb10_8, TX10_8, C10_8, A10_8, lamb11, TX11, C11, A11,...
%     lamb11_10, TX11_10, C11_10, A11_10, lamb11_9, TX11_9, C11_9, A11_9  );
% 
% [ lamb12_10, TX12_10, C12_10, A12_10 ] = interplant_7( nc+7, d, inx1, inx2, coef, lamb5, TX5, C5, A5,...
%     lamb5_4, TX5_4, C5_4, A5_4, lamb5_3, TX5_3, C5_3, A5_3��lamb6, TX6, C6, A6,...
%     lamb6_5, TX6_5, C6_5, A6_5, lamb6_4, TX6_4, C6_4, A6_4, lamb7, TX7, C7, A7,...
%     lamb7_6, TX7_6, C7_6, A7_6, lamb7_5, TX7_5, C7_5, A7_5, lamb8, TX8, C8, A8,...
%     lamb8_7, TX8_7, C8_7, A8_7, lamb8_6, TX8_6, C8_6, A8_6, lamb9, TX9, C9, A9,...
%     lamb9_8, TX9_8, C9_8, A9_8, lamb9_7, TX9_7, C9_7, A9_7, lamb10, TX10, C10, A10,...
%     lamb10_9, TX10_9, C10_9, A10_9, lamb10_8, TX10_8, C10_8, A10_8, lamb11, TX11, C11, A11,...
%     lamb11_10, TX11_10, C11_10, A11_10, lamb11_9, TX11_9, C11_9, A11_9  );
% % %.......

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


U1=V5-2*V5_4+V5_3;
U2=V6-2*V6_5+V6_4;
U3=V7-2*V7_6+V7_5;
U4=V8-2*V8_7+V8_6;
U5=V9-2*V9_8+V9_7;
U6=V10-2*V10_9+V10_8;
% U7=V11-2*V11_10+V11_9;
% U8=V12-2*V12_11+V12_10;

MuSIK(:,1) = U1;
MuSIK(:,2) = U1+U2;
MuSIK(:,3) = U1+U2+U3;
MuSIK(:,4) = U1+U2+U3+U4;
MuSIK(:,5) = U1+U2+U3+U4+U5;
MuSIK(:,6) = U1+U2+U3+U4+U5+U6;
% MuSIK(:,7) = U1+U2+U3+U4+U5+U6+U7;
% MuSIK(:,8) = U1+U2+U3+U4+U5+U6+U7+U8;

% f =@(X) (exp(-2*pi^2 .* X(:,1)) .* sin(pi .* X(:,2)) .* cos(pi .* X(:,3)));
% [ AP ] = f(TXY);

f =@(x) sin(pi*x(:,1)).*sin(pi*x(:,2)).*sin(pi*x(:,3));
[ AP ] = f(TXY);
[RMS, MAX] = deal(zeros(1,6));
for i = 1 : 6
    RMS(i) = rms( MuSIK(:,i) - AP );
    MAX(i) = max( abs( MuSIK(:,i) - AP ) );
end
