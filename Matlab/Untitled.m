syms theta4 theta5 theta6
R4 = [cos(theta4) -sin(theta4) 0;sin(theta4) cos(theta4) 0;0 0 1]*[0 1 0;0 0 1; 1 0 0];
R5 = [cos(theta5) -sin(theta5) 0;sin(theta5) cos(theta5) 0;0 0 1]*[0 1 0;0 0 1; 1 0 0];
R6 = [cos(theta6) -sin(theta6) 0;sin(theta6) cos(theta6) 0;0 0 1]*[0 -1 0;0 0 1;-1 0 0];
Ans = simplify(expand(R4*R5*R6))