function [ output ] = inner_test( t,x,lamb,TX,C,A )
% This is used in the PDE system re-construct for initial and boundary conditions
ch=length(TX);
V=ones(1,ch);
for j=1:ch
%   multiquadric RBF......
%     V1=  sqrt( ( (t-TX{j}(:,1)) ).^2 + (C{j}(1,1)./A{j}(1,1)).^2 );
%     V2=  sqrt( ( (x-TX{j}(:,2)) ).^2 + (C{j}(1,2)./A{j}(1,2)).^2 );
%     VV=V1.*V2;
%     V(j)=VV'*lamb{j};
%   .....................
%   Gaussian RBF  .......
    FAI1=  exp( -( A{j}(1,1)*(t-TX{j}(:,1)) ).^2 / C{j}(1,1)^2 );
    FAI2=  exp( -( A{j}(1,2)*(x-TX{j}(:,2)) ).^2 / C{j}(1,2)^2 );
    D = FAI1 .* FAI2;
    V(j)=D'*lamb{j};
%   .....................
end

output=sum(V);
end

