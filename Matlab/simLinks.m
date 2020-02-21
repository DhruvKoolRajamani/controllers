function dx = simLinks(x, x_d, Kp, Kv, Ki, step, ddq_d)
%SIMLINKS Summary of this function goes here
%   Detailed explanation goes here

global E
global p
global tau

M = [ (x(2,1) + 1/10)^2/20,    0,    0;
                         0, 1/20,    0;
                         0,    0, 1/20];

V = [                             0;
     -(x(4,1)^2*(x(2,1) + 1/10))/20;
                                  0];

G = [0; 0; -49/100];

e = x_d(1:3,1) - x(1:3,1);
de = x_d(4:6,1) - x(4:6,1);

M_d = [ (x_d(2,1) + 1/10)^2/20,    0,    0;
                             0, 1/20,    0;
                             0,    0, 1/20];

V_d = [                                 0;
       -(x_d(4,1)^2*(x_d(2,1) + 1/10))/20;
                                        0];

if p == 1
    E = E + e*step;
    u = Kp*e + Kv*de; % + Ki*E;
elseif p == 2
    u = M*(ddq_d + Kp*e + Kv*de) + V;
    tau = [tau u];
elseif p == 3                                  
    u = Kp*e + Kv*de + V_d + (M_d*ddq_d);
    tau = [tau u];
else
    u = 0;
end

dx(1,1) = x(4,1);
dx(2,1) = x(5,1);
dx(3,1) = x(6,1);
dx(4:6,1) = M\(u - V);
end

