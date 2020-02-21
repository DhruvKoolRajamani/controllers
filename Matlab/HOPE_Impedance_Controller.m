%% HOPE Impedance Controller
%
%% FK, IK, Dynamic Model Calculation
% 
clc, clear all, close all

syms l1 l2 m1 m2 q1 q2 dq1 dq2 pi g ddq1 ddq2 T1 T2 ...
     k1 k2 I1 I2 lc1 lc2
 
vals = 0;
% Uncomment these lines to view a plot of the arm with the axis
% 
l1 = 0.04297;
l2 = 0.04689;
m1 = 0.0049366;
m2 = 0.0034145;
g = 9.8;
lc1 = .023;
lc2 = .02071;
pi = double(pi);
% vals = [1.5708,-0.3,0].';

% system params
q = [q1 q2].';
dq = [dq1 dq2].';
ddq = [ddq1 ddq2].';
Tau = [T1 T2].';
m = [m1 m2].';

I1 = [4.4949e-07 -1.6594e-13 1.5842e-12;
          0        5.3509e-07  1.0268e-13;  
          0             0      2.2824e-07];
I2 = [1.5653e-07 -4.5517e-11 1.1444e-11;
           0      2.2255e-7  -1.9381e-8;
           0          0       1.1827e-07];
% dh params
thetas = [q1 q2].';
d = [0 0].';
a = [-l1 -l2].';
alpha = [0 0].';
[n_dof,rand] = size(q);

% vals = [0, 0.1].';
%%% Calculating the Forward Kinematics for each link
%
% <<plotArm.m>>
% Transforms - T01,...,T0n
[Transforms, T0n] = plotArm(q,thetas,d,a,alpha,vals,[0, 0, 90],true);
if size(vals) > [1,1]
    axis([-1 1 -1 1 -1 1]);
    drawnow;
end
T0n

%%% Calculating the Jacobians
% 
% <<calcJacobian.m>>
% Jv = linear velocity, Jw = angular velocity, p = position vector, z = approach vector

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

l1 = 0.04297;
l2 = 0.04689;
m1 = 0.0049366;
m2 = 0.0034145;
g = 9.8;
lc1 = .023;
lc2 = .02071;

q = [q1 q2].';
dq = [dq1 dq2].';
x = [q; dq];

M = vpa(subs(M),5);
M = vpa(subs(M,[pi;q;dq],[double(pi);x]),5)
V = vpa(subs(V),5);
V = vpa(subs(V,[pi;q;dq],[double(pi);x]),5)
G = vpa(subs(G),5);
G = vpa(subs(G,[pi;q;dq],[double(pi);x]),5)

Tau_Stiffness_Relaxed = [1.118e-18*dq1^5 - 1.4724e-14*dq1^4 + 7.2021e-11*dq1^3 - 1.5778e-7*dq1^2 + 0.00010758*dq1 + 0.14548;
                 3.6689e-9*dq2^5 - 6.6466e-7*dq2^4 + 0.000043944*dq2^3 - 0.001267*dq2^2 + 0.013425*dq2 + 0.03056];

% Tau_stiffness = [TAU_MCP; TAU_PIP];

% Tau = vpa(M*ddq + V + G + Tau_stiffness);

syms u1 u2
u = [u1; u2];
dx = sym(zeros(4,1));
dx(1,1) = dq1;
dx(2,1) = dq2;
dx(3:4,1) = vpa(M\(u - V - Tau_Stiffness_Relaxed),3);

%% Straight Line Trajectory + Returning Trajectory
%

global p
global tau
global tm1
global err
global alpha
global act_tau
p = 4;
tau = [];
tm1 = [];
err = [];
act_tau = [];

Kp = 1*[0, 0; 0, 1];
Kv = 0.01*Kp;

tf1 = 5;

syms t_

% vec_t = [0.6982/5; 0.4364/5];
vec_t = [0.4364/5; 0.4364/5];
pos_d(1:2,1) =   vec_t*(sin(10*t_)+1);
% dvec_t = [3.49/5; 2.18/5];
dvec_t = [2.18/5; 2.18/5];
vel_d(1:2,1) = 2*dvec_t*cos(10*t_);
% ddvec_t = [-17.5/5; -10.9/5];
ddvec_t = [-10.9/5; -10.9/5];
acc_d(1:2,1) = 2*ddvec_t*sin(10*t_);

q_init = double(subs(pos_d,t_,0))
dq_init = double(subs(vel_d,t_,0))
ddq_init = double(subs(acc_d,t_,0))
x_init = [q_init;dq_init];
q_d = double(subs(pos_d,t_,tf1))
dq_d = double(subs(vel_d,t_,tf1))
ddq_d = double(subs(acc_d,t_,tf1))
x_d = [q_d; dq_d];

trajectory_coefficients = [vec_t; dvec_t; ddvec_t];

L = 50*eye(12);
% [a0 + a1x + a2x^2 + a3x^3]
alpha0 = [0.1; 0.01; 0.001; 0.0001; 0.00001; 0.000001; 0.1; 0.01; 0.001; 0.0001; 0.00001; 0.000001];
% alpha0 = [0; 0; 0; 0; 0; 0; 0; 0; 0; 0; 0; 0; 0; 0];
% alpha0 = [0; 0; 0; 0; 0; 0; 0];

% [T1,X1] = ode15s(@(t,x)simArm(x, x_d, Kp, Kv, ddq_d, cart_des_1_traj, t, 1, [0 tf1]),[0 tf1],x_init);
options = odeset('RelTol',1e-4,'AbsTol',[1e-4, 1e-4, 1e-4, 1e-4]);
[T1,X1] = ode15s(@(t,x)simAdaptiveArm(x, trajectory_coefficients, x_d, Kp, Kv, L, t, alpha0, ddq_init),[0 tf1],x_init,options);
alpha
T = T1;
X = X1;

u = tau.';
au = act_tau.';
Err = err.';

%%
[mT, nT] = size(T);
[mtm1, ntm1] = size(tm1);
tm = tm1;
t_ = [0:0.01:tf1].';
traj_d(:,1:2) = subs(pos_d.');
vtraj_d(:,1:2) = subs(vel_d.');
[mtm, ntm] = size(tm);
x1 = X(:,1);
x2 = X(:,2);
QD1 = zeros(mT,1);
QD1(:,1) = 1.3963;
QD2 = zeros(mT,1);
QD2(:,1) = 0.8727;
x0 = zeros(mT,1);
dx1 = X(:,3);
dx2 = X(:,4);
u0 = zeros(mtm,1);
u1 = u(1:mtm,1);
u2 = u(1:mtm,2);
au0 = zeros(mtm,1);
au1 = au(1:mtm,1);
% au2 = au(1:mtm,2);
% theta = zeros(mtm,1);
% for i=1:mtm
%     theta(i,1) = atan2(au1(i,1),au2(i,1));
% end
e0 = zeros(mtm,1);
e1 = Err(1:mtm,1);
e2 = Err(1:mtm,2);

figure(1);
% subplot(2,2,1);
plot(T,x1, '-r','DisplayName', 'MCP adaptive trajectory');
hold on;
plot(t_,traj_d(:,1),'--r','DisplayName', 'MCP desired trajectory');
plot(T,x2, '-b','DisplayName', 'PIP adaptive trajectory');
plot(t_,traj_d(:,2),'--b','DisplayName', 'PIP desired trajectory');
plot(T,x0);
hold off
xlabel({'Time','(0 \leq t \leq 10)'})
ylabel({'Angular Positions','(rads)'})
legend('MCP adaptive trajectory', 'MCP desired trajectory', 'PIP adaptive trajectory', 'PIP desired trajectory')
title('Joint angle vs Time');
% subplot(2,2,2);
figure(2);
plot(T,dx1, '-r','DisplayName', 'MCP adaptive velocities');
hold on;
plot(t_,vtraj_d(:,1),'--r','DisplayName', 'MCP desired velocities');
hold on;
plot(T,dx2, '-b','DisplayName', 'PIP adaptive velocities');
plot(t_,vtraj_d(:,2),'--b','DisplayName', 'PIP desired velocities');
plot(T,x0);
hold off
xlabel({'Time','(0 \leq t \leq 10)'})
ylabel({'Angular Velocities','(rads/seconds)'})
legend('MCP adaptive velocities', 'MCP desired velocities', 'PIP adaptive velocities', 'PIP desired velocities')
title('Joint velocities vs Time');
% subplot(2,2,3);
figure(3);
plot(tm,u1, '-r', 'DisplayName','MCP adaptive torques');
hold on;
plot(tm,u2, '-b', 'DisplayName','PIP adaptive torques');
hold on;
plot(tm,u0);
hold off
xlabel({'Time','(0 \leq t \leq 10)'})
ylabel({'Torque','(Nm)'})
legend('MCP adaptive torques', 'PIP adaptive torques')
title('Inputs vs Time');
figure(4);
% subplot(2,2,4);
plot(tm,e1);
hold on;
plot(tm,e2);
hold on;
plot(tm,e0);
title('Errors vs Time');
figure(5);
plot(tm,au1, '-r', 'DisplayName','adaptive torque error');
hold on;
% plot(tm,au2, '-b', 'DisplayName','PIP adaptive torques');
% hold on;
plot(tm,au0);
hold off
xlabel({'Time','(0 \leq t \leq 10)'})
ylabel({'Torque','(Nm)'})
legend('adaptive torque error')
title('Inputs vs Time');

% for i=1:length(T)
%     gcf;
%     figure(2);
% %     subplot(2,1,1)
%     vals(1) = X(i,1);
%     vals(2) = X(i,2);
%     g = 9.8;
%     q = [q1 q2];
%     pi = double(pi);
%     p_ = subs(p02);
%     p_ = subs(p_,q,vals);
%     drawManip(p_,[0 0 90]);
%     view([0 0 90]);
%     axis([-0.25 0.25 -0.25 0.25 -0.25 0.25]);
%     drawnow;
%     hold off
% end