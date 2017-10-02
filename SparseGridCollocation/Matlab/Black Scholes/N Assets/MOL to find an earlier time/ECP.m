function [ P ] = ECP( X, r, sigma, T, E)

t=X(:,1);
S=X(:,2);
M=T-t;
[N,~] = size(X);
P = ones(N,1);
d1 = ones(N,1);
d2 = ones(N,1);

I0 = M == 0;
I1 = M ~= 0;

P(I0) = max( S(I0) - E, 0);

d1(I1)=(log(S(I1) ./ E) + (r+sigma^2/2) .* M(I1)) ./ (sigma .* sqrt(M(I1)));
d2(I1)=(log(S(I1) ./ E) + (r-sigma^2/2) .* M(I1)) ./ (sigma .* sqrt(M(I1)));
P(I1) = -E .* exp(-r .* M(I1)) .* normcdf(d2(I1)) + S(I1) .* normcdf(d1(I1));

end

