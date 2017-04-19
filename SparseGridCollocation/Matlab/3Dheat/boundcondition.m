function [ Value ] = boundcondition( X )
% Initial condition and boundary condition used for particular target

% # f = exp(-2*pi^2 .* t) .* sin(pi .* x) .* cos(pi .* y) # target function

% Value = exp(-2*pi^2 .* X(1)) .* sin(pi .* X(2)) .* cos(pi .* X(3));

% f = sin(pi*x1)*sin(pi*x2)*sin(pi*t)

Value = sin(pi*X(1)) * sin(pi*X(2)) * sin(pi*X(3));

end

