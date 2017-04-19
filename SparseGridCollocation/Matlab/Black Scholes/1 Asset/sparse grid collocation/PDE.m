function [ output ] = PDE( node,r,sigma,lamda2,TX2,C2,A2,lamda3,TX3,C3,A3  )
% This is used in PDE system re-construct for PDE
[N,~] = size(node);
ch2=length(TX2);
U2=ones(N,ch2);
for j=1:ch2
    
    [ FAI2, FAI2_t, FAI2_x, FAI2_xx ] = mq2d( node, TX2{j}, A2{j}, C2{j} );
%   this equation is determined specially by B-S
    U2(:,j) = FAI2_t*lamda2{j} + sigma^2/2*FAI2_xx*lamda2{j} + r*FAI2_x*lamda2{j} - r*FAI2*lamda2{j};
end

ch3=length(TX3);
U3=ones(N,ch3);
for j=1:ch3
    
    [ FAI3, FAI3_t, FAI3_x, FAI3_xx ] = mq2d( node, TX3{j}, A3{j}, C3{j} );
%   this equation is determined specially by B-S
    U3(:,j) = FAI3_t*lamda3{j} + sigma^2/2*FAI3_xx*lamda3{j} + r*FAI3_x*lamda3{j} - r*FAI3*lamda3{j};
end
% output is depending on the combination tech
output=(sum(U3,2) - sum(U2,2));
