function [ U ] = PayoffFunction( underlyingPrices, Strike )
%Compute the S-term in an option payoff function 

S = PayoffS(underlyingPrices);
K = zeros(size(S));
U = max(S-Strike, K);

end

