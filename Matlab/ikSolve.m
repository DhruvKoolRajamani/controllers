function [qd] = ikSolve(cart)
%IKSOLVE Summary of this function goes here
%   Detailed explanation goes here
syms t_

l1 = 0.04297;
l2 = 0.04689;

x = cart(1);
y = cart(2);

qd2 = acos((x^2 + y^2 - l1^2 - l2^2)/(2*l1*l2));
qd1 = atan2(x,-y) - atan2((l2*sin(qd2)),(l1 + l2*cos(qd2)));

qd = [qd1; qd2];

end

