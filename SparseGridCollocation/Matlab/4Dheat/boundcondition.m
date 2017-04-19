function [ Value ] = boundcondition( X )
% Initial condition and boundary condition used for particular target

% # f = sin(pi*t) * sin(pi*x) * sin(pi*y) * sin(pi*z) # target function

Value = sin(pi*X(1)) * sin(pi*X(2)) * sin(pi*X(3)) * sin(pi*X(4));

end

