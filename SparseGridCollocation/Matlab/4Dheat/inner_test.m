function [ output ] = inner_test( X,lamb,TX,C,A )
% This is used in the PDE system re-construct for initial and boundary conditions
ch=length(TX);
V=ones(1,ch);
for j=1:ch
% multiquadric RBF.........
% V1=  sqrt( ( (X(1)-TX{j}(:,1)) ).^2 + (C{j}(1,1)./A{j}(1,1)).^2 );
% V2=  sqrt( ( (X(2)-TX{j}(:,2)) ).^2 + (C{j}(1,2)./A{j}(1,2)).^2 );
% V3=  sqrt( ( (X(3)-TX{j}(:,3)) ).^2 + (C{j}(1,3)./A{j}(1,3)).^2 );
% V4=  sqrt( ( (X(4)-TX{j}(:,4)) ).^2 + (C{j}(1,4)./A{j}(1,4)).^2 );
% 
% VV=V1.*V2.*V3.*V4;
% 
% V(j)=VV'*lamb{j};
% ..........................
% Gaussian RBF..........
V1=  exp( -( A{j}(1,1)*(X(1)-TX{j}(:,1)) ).^2 / C{j}(1,1)^2 );
V2=  exp( -( A{j}(1,2)*(X(2)-TX{j}(:,2)) ).^2 / C{j}(1,2)^2 );
V3=  exp( -( A{j}(1,3)*(X(3)-TX{j}(:,3)) ).^2 / C{j}(1,3)^2 );
V4=  exp( -( A{j}(1,4)*(X(4)-TX{j}(:,4)) ).^2 / C{j}(1,4)^2 );
VV=V1.*V2.*V3.*V4;

V(j)=VV'*lamb{j};
% .........................
end

output=sum(V);
end

