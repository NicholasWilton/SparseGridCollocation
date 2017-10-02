function Cartesian = CartesianProduct(grid)

[~,numAssets] = size(grid);

[F{1:numAssets}] = ndgrid(grid(:));

[i,j] = size(F{1});
G = zeros(i*j,numAssets);
for i=numAssets:-1:1
    G(:,i) = F{i}(:);
end

Cartesian = unique(G , 'rows');
end

