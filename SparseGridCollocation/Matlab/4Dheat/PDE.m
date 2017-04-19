function [ output ] = PDE( node,lamda1,TX1,C1,A1,lamda2,TX2,C2,A2,lamda3,TX3,C3,A3,...
    lamda4,TX4,C4,A4)

[N,~] = size(node);
ch=length(TX1);
U1=ones(N,ch);
for j=1:ch
    [ ~, FAI_t, FAI_xx, FAI_yy, FAI_zz ] = mq4d( node, TX1{j}, A1{j}, C1{j} );
%   this equation is determined specially by different PDE
    U1(:,j) = FAI_t*lamda1{j} - FAI_xx*lamda1{j} - FAI_yy*lamda1{j} - FAI_zz*lamda1{j};
end

ch=length(TX2);
U2=ones(N,ch);
for j=1:ch

    [ ~, FAI_t, FAI_xx, FAI_yy, FAI_zz ] = mq4d( node, TX2{j}, A2{j}, C2{j} );

    U2(:,j) = FAI_t*lamda2{j} - FAI_xx*lamda2{j} - FAI_yy*lamda2{j} - FAI_zz*lamda2{j};
end


ch=length(TX3);
U3=ones(N,ch);
for j=1:ch

    [ ~, FAI_t, FAI_xx, FAI_yy, FAI_zz ] = mq4d( node, TX3{j}, A3{j}, C3{j} );

    U3(:,j) = FAI_t*lamda3{j} - FAI_xx*lamda3{j} - FAI_yy*lamda3{j} - FAI_zz*lamda3{j};
end

ch=length(TX4);
U4=ones(N,ch);
for j=1:ch

    [ ~, FAI_t, FAI_xx, FAI_yy, FAI_zz ] = mq4d( node, TX4{j}, A4{j}, C4{j} );

    U4(:,j) = FAI_t*lamda4{j} - FAI_xx*lamda4{j} - FAI_yy*lamda4{j} - FAI_zz*lamda4{j};
end
% output is depending on the combination tech
output=sum(U1,2)-3*sum(U2,2)+3*sum(U3,2)-sum(U4,2);
