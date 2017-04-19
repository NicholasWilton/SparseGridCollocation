function [ D, Dt, Dx, Dxx ] = mq2d( TP, CN, A, C )
% For B-S
% TP for test points, CN for center nodes. 
% First column represents time dimension, Second column represents space dimension
% A is scale parameter
% C is shape parameter

[Num,~]=size(CN);
[N,~]=size(TP);
[D,Dt,Dx,Dxx] = deal(ones(N,Num));
% multiquadric RBF..............
% for j=1:Num
%     FAI1=  sqrt( (C(1,1)/A(1,1))^2 + (TP(:,1)-CN(j,1)).^2 );
%     FAI2=  sqrt( (C(1,2)/A(1,2))^2 + (TP(:,2)-CN(j,2)).^2 );
%     D(:,j)=FAI1.*FAI2;
%     if nargout > 1
%        Dt(:,j) = (TP(:,1)-CN(j,1)).*FAI2./FAI1;
%        Dx(:,j) = TP(:,2) .* ( (TP(:,2)-CN(j,2)).*FAI1./FAI2 );
%        Dxx(:,j) = TP(:,2).^2 .* (FAI1./FAI2 - FAI1.*(TP(:,2)-CN(j,2)).^2./FAI2.^3);
%     end
% end
% .........................
% Gaussian RBF.............
for j=1:Num
    a =( A(1,1)*(TP(:,1)-CN(j,1)) );
    FAI1=  exp( -a.^2 / C(1,1)^2 );
    FAI2=  exp( -( A(1,2)*(TP(:,2)-CN(j,2)) ).^2 / C(1,2)^2 );
    D(:,j)=FAI1.*FAI2;
    if nargout > 1
       Dt(:,j) = -2*(A(1,1)/C(1,1))^2 * (TP(:,1)-CN(j,1)) .* FAI1 .* FAI2;
       Dx(:,j) = TP(:,2).*(-2*(A(1,2)/C(1,2))^2 * (TP(:,2)-CN(j,2)) .* FAI1 .* FAI2);
       Dxx(:,j)= TP(:,2).^2.*(( -2*A(1,2)^2/C(1,2)^2 + 4*A(1,2)^4*(TP(:,2)-CN(j,2)).^2./C(1,2)^4 ) .* FAI2 .* FAI1);
    end
end
% .........................
end

