function [Kp, Kv, Ki] = calcGains(G,q)
%CALCGAINS Summary of this function goes here
%   Detailed explanation goes here
syms m1 m2 I1 I2 l1 lc1 lc2 q1 q2 g
dG = sym(zeros(2,2));
for i=1:2
    for j=1:2
        dG(i,j) = diff(G(i), q(j));
    end
end

% The maximum value is the dG(1,1) when q1 = q2 = 0

dG_vals = double(subs(dG, [l1, lc1, lc2, m1, m2, I1, I2, g, q1, q2], ...
      [0.26, 0.0983, 0.0229, 6.5225, 2.0458, 0.1213, 0.0116, 9.81, 0, 0]));

n=2;
Kg = n*max(max(dG_vals).');
lm_min_M = 0.011;
lm_max_M = 0.361;

Ki = 1.5*diag([1,1]);
Kp = 30*diag([1,1]);
Kv = diag([7,3]);

end

