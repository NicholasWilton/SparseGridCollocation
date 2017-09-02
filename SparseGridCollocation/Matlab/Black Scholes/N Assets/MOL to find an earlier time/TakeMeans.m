function [ N ] = TakeMeans( M )
%TAKEMEANS reduce M to means in Col1 of duplicates in Col2 where Col2 is
%sorted in ascending order

[I,J] = size(M);

last = -realmax;
sum =0;
Ni =0;
Na = zeros(I,J);
count =0;
for i=1:I
    current = M(i,2);
   
    if (i ~= 1 && current ~= last)
     Ni = Ni +1;
     Na(Ni,1) = M(i-1,1);
     Na(Ni,2) = sum / count;
     count =1;
     sum = current;
    else
        count = count +1;
        sum = sum + current;
    end
    last = current;
end

N = Na(1:Ni,1:2);

end

