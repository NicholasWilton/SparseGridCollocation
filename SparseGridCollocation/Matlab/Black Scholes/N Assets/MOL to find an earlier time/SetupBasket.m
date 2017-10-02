function [ inx1, inx2, TestNodes, CentralNodes, AroundStrikeNodes ] ...
= SetupBasket( numAssets, TestNodeNumber, CentralNodeNumber, AroundStrikeRange, Strike )

inx1 = -1 * ones([1,numAssets]) * Strike;
inx2 =  2 * ones([1,numAssets]) * Strike;

testGrid = deal(zeros([TestNodeNumber, numAssets]));
for i=1:numAssets
    testGrid(:,i) = linspace(inx1(1,i),inx2(1,1),TestNodeNumber)';
end

TestNodes = CartesianProduct(testGrid);

centralGrid = deal(zeros(CentralNodeNumber, numAssets));
for i=1:numAssets
    centralGrid(:,i) = linspace(0,3*Strike,CentralNodeNumber)';
end
CentralNodes = CartesianProduct(centralGrid);

IT = testGrid >= (1-AroundStrikeRange)*Strike & testGrid <= (1 + AroundStrikeRange)*Strike;
%IT will be a logical matrix with non-zero elements making up a rectangular
%sub section. We need that subsection:
iStart = -1;
iLen = -1;
jStart =-1;
jLen =-1;

[ITi, ITj] = size(IT);
for i=1:ITi
   for j=1:ITj
    if IT(i,j) > 0 && (iStart < 0 && jStart < 0)
        iStart = i;
        jStart = j;
    end
    if IT(i,j) < 1 && (iStart > 0 && jStart > 0) && (iLen < 0  && jLen < 00)
            iLen = i - iStart;
            [~,jLen] = size(IT);
    end
        
   end 
end
aroundStrikeGrid = zeros(iLen, jLen);
x=1;
y=1;
for i=iStart:iStart+iLen -1
    for j=jStart:jStart+jLen -1
         aroundStrikeGrid(x,y) = testGrid(i,j);
        y=y+1;
    end
    y=1;
    x = x+1;
end

AroundStrikeNodes = CartesianProduct(aroundStrikeGrid);

end

