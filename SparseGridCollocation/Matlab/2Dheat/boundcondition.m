function [ Value ] = boundcondition( X )
% Initial condition and boundary condition used for particular target

% # f = exp(-pi^2*t)*sin(pi*x) # target function
Value = exp(-pi^2*X(1)) * sin(pi * X(2));

end

