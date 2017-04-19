function [phi,phi1,phi2,phi3] = mq2d(x,xc,c)
% 1-D multiquadric radial basis function
f = @(r,c) sqrt(r.^2 + c.^2);

r = x - xc;
phi = f(r,c);

if nargout > 1
% 1-st derivative    
phi1 = x.*r./phi;
    if nargout > 2
    % 2-nd derivative
    phi2 = x.^2.*(c^2)./(phi.^3);
    if nargout > 3
%         3-rd derivative
        phi3 = -3*c^2.*r./(phi.^5);
    end
    end
end
% 1-D Gaussian radial basis function
% f = @(r,c) exp(-r.^2 ./ c.^2);
% 
% r = x - xc;
% phi = f(r,c);
% 
% if nargout > 1
% % 1-st derivative    
% phi1 = x.*(-2*r./c.^2.*phi);
%     if nargout > 2
%     % 2-nd derivative
%     phi2 = x.^2.*((4*r.^2-2*c.^2)./c.^4.*phi);
%     if nargout > 3
% %         3-rd derivative
%         phi3 = (12*r.*c.^2-8*r.^3)./c.^6.*phi;
%     end
%     end
% end