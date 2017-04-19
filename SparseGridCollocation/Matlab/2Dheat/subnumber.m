function [ L ] = subnumber( b, d )
% Find possible layouts L that satisfy |L|_1 = b
% [~,N] = size(L), N = d.
if d==1 
    L(1)=b;
else
    nbot=1;
    for i=1:b-d+1
        indextemp=subnumber(b-i, d-1);
        [s,~]=size(indextemp);
        ntop=nbot+s-1;
        L(nbot:ntop,1)=i*ones(s,1);
        L(nbot:ntop,2:d)=indextemp;
        nbot=ntop+1;
    end
end

end

