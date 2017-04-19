function [ D, Dt, Dxx, Dyy ] = mq3d( TP, CN, A, sp )
% For heat equation
% TP for test points, CN for center nodes
% First column represents time dimension, other two columns represent space dimensions
% A is scale parameter
% sp is shape parameter

[Num,~]=size(CN);
[N,~]=size(TP);
[D,Dt,Dxx,Dyy] = deal(ones(N,Num));
% multiquadric RBF........
% for j=1:Num
%     FAI1=  sqrt( (sp(1,1)/A(1,1))^2 + (TP(:,1)-CN(j,1)).^2 );
%     FAI2=  sqrt( (sp(1,2)/A(1,2))^2 + (TP(:,2)-CN(j,2)).^2 );
%     FAI3=  sqrt( (sp(1,3)/A(1,3))^2 + (TP(:,3)-CN(j,3)).^2 );
%     D(:,j)=FAI1.*FAI2.*FAI3;
%     if nargout > 1
%        Dt(:,j) = (TP(:,1)-CN(j,1)).*FAI2.*FAI3./FAI1;
%        Dxx(:,j)= ( 1./FAI2 - (TP(:,2)-CN(j,2)).^2./FAI2.^3 ) .* FAI1 .* FAI3;
%        Dyy(:,j)= ( 1./FAI3 - (TP(:,3)-CN(j,3)).^2./FAI3.^3 ) .* FAI1 .* FAI2;
%     end
% end
% ...................
% Gaussian RBF...........
for j=1:Num      
    FAI1=  exp( -( A(1,1)*(TP(:,1)-CN(j,1)) ).^2 / sp(1,1)^2 );
    FAI2=  exp( -( A(1,2)*(TP(:,2)-CN(j,2)) ).^2 / sp(1,2)^2 );
    FAI3=  exp( -( A(1,3)*(TP(:,3)-CN(j,3)) ).^2 / sp(1,3)^2 );
    D(:,j)=FAI1.*FAI2.*FAI3;
    if nargout > 1
       Dt(:,j) = -2*(A(1,1)/sp(1,1))^2 * (TP(:,1)-CN(j,1)) .* FAI1 .* FAI2 .* FAI3;
       Dxx(:,j)= ( -2*A(1,2)^2/sp(1,2)^2 + 4*A(1,2)^4*(TP(:,2)-CN(j,2)).^2./sp(1,2)^4 ) .* FAI2 .* FAI1 .*FAI3;
       Dyy(:,j)= ( -2*A(1,3)^2/sp(1,3)^2 + 4*A(1,3)^4*(TP(:,3)-CN(j,3)).^2./sp(1,3)^4 ) .* FAI2 .* FAI1 .*FAI3;
    end
end
% ...................
end

