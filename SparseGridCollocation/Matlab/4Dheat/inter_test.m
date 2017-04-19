function [ output ] = inter_test( X, lamb, TX, C, A )
% This is used to calculate values on final testing points
ch=length(TX);
[M,~]=size(X);
V=ones(M,ch);
for j=1:ch

[ D ] = mq4d( X, TX{j}, A{j}, C{j} );
V(:,j)=D*lamb{j};

end
output=sum(V,2);
end

