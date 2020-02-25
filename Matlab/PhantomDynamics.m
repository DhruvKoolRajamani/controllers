clc; clear all; close all;

syms q1 q2 dq1 dq2 pi g ddq1 ddq2 T1 T2
I1 = [1.619e-5,  4.46e-6,    7e-8;
          4.46e-6,  7.39e-6,  1.9e-7;
             7e-8,   1.9e-7,  1.7e-5];
m1 = 0.03955516;
m2 = 0.03228749;
l1 = 0.043;
lc1 = sqrt(0.011^2 + 0.00512716);
l2 = 0.05604;
lc2 = sqrt(0.01690504^2 + 0.00466182^2);

I2 = [4.19e-6, 4.87e-6, -1e-8;
         4.87e-6, 2.065e-5, -1e-8;
         -1e-8, -1e-8, 2.17e-5];
     
g = 9.8;
pi = double(pi);
% system params
q = [q1 q2].';
dq = [dq1 dq2].';
ddq = [ddq1 ddq2].';
Tau = [T1 T2].';
m = [m1 m2].';

thetas = [q1 q2].';
d = [0, 0].';
a = [l1, l2].';
al = [0, 0].';
[n_dof,rand] = size(q);
vals = 0;
% vals = [-20*pi/180, -40*pi/180].';
[Transforms, T0n] = plotArm(q,thetas,d,a,al,vals,[0, 0, 90],true);
if size(vals) > [1,1]
    axis([-1 1 -1 1 -1 1]);
    drawnow;
end
T0n

[Jv, Jw, p02, z] = calcJacobian(Transforms, q, 'rr', vals);
J = [Jv(:,:,n_dof); Jw(:,:,n_dof)];
reduced_J = J(1:2,:);
% reduced_J(3,:) = J(6,:);

CoM_Transforms = Transforms;
CoM_Transforms(:,:,1) = (subs(CoM_Transforms(:,:,1),l1,lc1));
CoM_Transforms(:,:,2) = (subs(CoM_Transforms(:,:,2),l2,lc2));

[Jv_CoM, Jw_CoM, p_CoM, z_CoM] = calcJacobian(CoM_Transforms, q, 'rr', vals);
% substitute center of mass coordinates

J_CoM = [Jv_CoM(:,:,n_dof); Jw_CoM(:,:,n_dof)];

%%% Calculating the Kinetic and Potential Energies
% This hasn't been automated to keep for correctness
% Will need to modify this to fit our design (Use the inertia matrix
% instead of M provided by the solidworks model)

KE = vpa(0.5*dq.'*(m(1,1)*Jv_CoM(:,:,1).'*Jv_CoM(:,:,1) + Jw_CoM(:,:,1).'*CoM_Transforms(1:3,1:3,1)*I1*CoM_Transforms(1:3,1:3,1).'*Jw_CoM(:,:,1) + ... % link 1 translational and rotational KE
                m(2,1)*Jv_CoM(:,:,2).'*Jv_CoM(:,:,2)+ Jw_CoM(:,:,2).'*CoM_Transforms(1:3,1:3,2)*I2*CoM_Transforms(1:3,1:3,2).'*Jw_CoM(:,:,2))*dq);   % link 2 translational and rotational KE
PE = vpa(m(1,1)*g*p_CoM(2,2) + m(2,1)*g*p_CoM(2,3));

%%% 
% Euler Lagrange Equation:
% 
% $$F_{i} = \frac{\mathrm{d} }{\mathrm{d} t}\frac{\partial L}{\partial \dot{q_{i}}} - \frac{\partial L}{\partial q_{i}}$$
% 

L = KE - PE;

% Using the chain rule:
% dLdq1 = diff(L,dq1);
% ddLdq1 = diff(dLdq1,dq1)*ddq1 + diff(dLdq1,q1)*dq1 + ... 
%          diff(dLdq1,dq2)*ddq2 + diff(dLdq1,q2)*dq2 + ...
%          diff(dLdq1,dq3)*ddq3 + diff(dLdq1,q3)*dq3;
% dLdq2 = diff(L,dq2);
% ddLdq2 = diff(dLdq2,dq1)*ddq1 + diff(dLdq2,q1)*dq1 + ... 
%          diff(dLdq2,dq2)*ddq2 + diff(dLdq2,q2)*dq2 + ...
%          diff(dLdq2,dq3)*ddq3 + diff(dLdq2,q3)*dq3;
% dLdq3 = diff(L,dq3);
% ddLdq3 = diff(dLdq3,dq1)*ddq1 + diff(dLdq3,q1)*dq1 + ... 
%          diff(dLdq3,dq2)*ddq2 + diff(dLdq3,q2)*dq2 + ...
%          diff(dLdq3,dq3)*ddq3 + diff(dLdq3,q3)*dq3;
% dLq1 = diff(L,q1);
% dLq2 = diff(L,q2);
% dLq3 = diff(L,q3);
% 
% Tau1 = ddLdq1 - dLq1;
% Tau2 = ddLdq2 - dLq2;
% Tau3 = ddLdq3 - dLq3;
% Tau = [Tau1; Tau2; Tau3];

% This function does n joints in a for loop
for i=1:n_dof
    dLdq(i,1) = simplify(diff(L,dq(i,1)));
    for j=1:n_dof
        ddLdq_mat(i,j) = simplify(diff(dLdq(i,1),dq(j,1))*ddq(j,1) + diff(dLdq(i,1),q(i,1))*dq(i,1));
    end
    ddLdq(i,1) = sum(ddLdq_mat(i,:));
    dLq(i,1) = diff(L,q(i,1));
    Tau(i,1) = ddLdq(i,1) - dLq(i,1);
end
Tau1 = Tau;

% Calculating intertial matrix M by substituting ddq as 0 and dividing by ddq
for i=1:n_dof
    for j=1:n_dof
        M(i,j) = simplify(expand(Tau(i,1) - subs(Tau(i,1), ddq(j,1), 0))/ddq(j,1));
    end
end

% Calculating gravity matrix G
G = subs(Tau, [ddq;dq], [0;0;0;0]);

% Calculating the Coriolis and Centrifugal Terms V
for i=1:n_dof
     V(i,1) = simplify(expand(Tau(i,1) - M(i,:)*ddq - G(i)));
end

Tau = M*ddq + V + G; % Dynamic Model

q = [q1 q2].';
dq = [dq1 dq2].';
x = [q; dq];

M = vpa(subs(M),5);
M = vpa(subs(M,[pi;q;dq],[double(pi);x]),5)
V = vpa(subs(V),5);
V = vpa(subs(V,[pi;q;dq],[double(pi);x]),5)
G = vpa(subs(G),5);
G = vpa(subs(G,[pi;q;dq],[double(pi);x]),5)
