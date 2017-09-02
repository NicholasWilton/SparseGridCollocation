function [ S ] = PayoffS( underlyingPrices )

S = mean(underlyingPrices, 2);

end

