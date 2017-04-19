function [ output ] = inner_test( X,lamb,TX,C,A )
% This is used in the PDE system re-construct for initial and boundary conditions
ch=length(TX);
V=ones(1,ch);
% multiquadric RBF........
% for j=1:ch
%     FAI1=  sqrt( (C{j}(1,1)/A{j}(1,1))^2 + (X(1,1)-TX{j}(:,1)).^2 );
%     FAI2=  sqrt( (C{j}(1,2)/A{j}(1,2))^2 + (X(1,2)-TX{j}(:,2)).^2 );
%     D = FAI1.*FAI2;
%     V(j) = D' * lamb{j};
% end
% .......................
% Gaussian RBF...........
for j=1:ch
    V1 =  exp( -( A{j}(1,1).*(X(1)-TX{j}(:,1)) ).^2 ./ C{j}(1,1)^2 );
    V2 =  exp( -( A{j}(1,2).*(X(2)-TX{j}(:,2)) ).^2 ./ C{j}(1,2)^2 );
    VV=V1.*V2;
    V(j) = VV'*lamb{j};
end
% ......................
output=sum(V);
end

