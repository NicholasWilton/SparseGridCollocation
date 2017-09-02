function [ dx ] = DiffN( x )
%Calculates diff in multiple dimensions, i.e. euclidean distance between
%subsequent nodes
[I,~] = size(x);
dx=zeros(I-1,1);
for i=1:I-1
    dx(i,1) = (sum( (x(i+1,:) - x(i,:)).^2) ).^(1/2);
end

