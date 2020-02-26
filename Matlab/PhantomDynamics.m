clc; clear all; close all;

syms q1 q2 dq1 dq2 pi g ddq1 ddq2 T1 T2
% I1 = [8.53e-6,  5.6e-7,    -1.5e-7;
%           5.6e-7,  2.26e-5,  -1.6e-7;
%              -1.5e-7,   -1.6e-7,  1.883e-5];
m1 = 0.05704916;
m2 = 0.04625979;
l1 = 0.043;
lc1 = sqrt(0.00960803^2 + 0.00028241);
l2 = 0.05604;
lc2 = sqrt(0.05195307^2 + 0.00944479^2);

% I2 = [1.434e-5, 2.883e-5, -1.3e-7;
%          2.883e-5, 1.36e-4, 4e-8;
%          -1.3e-7, 4e-8, 1.44e-4];
     
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
simplify(expand(Transforms(1:3,4,:)))

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

I1 = 0.00001355 + m1*lc1^2;
I2 = 0.00001502 + m2*lc2^2;

% c1 = m1*lc1^2 + m2*l1^2 + I1;
% c2 = m2*lc2^2;
% c3 = m2*l1*lc2;
% c4 = (m1*lc1 + m2*l1)*g;
% c5 = m2*g*lc2;
% 
% M = [c1+c2-2*c3*cos(q2), c2-c3*sin(q2);
%           c2-c3*sin(q2),           c2];
% 
% C = [ c3*dq2*sin(q2), c3*(dq1 + dq2)*sin(q2);
%      -c3*dq1*sin(q2),                    0 ];
% 
% G = [c4*cos(q1) - c5*cos(q1+q2);
%      -c5*cos(q1+q2)];

M = [I1 + I2 + m2*l1^2 + 2*m2*l1*lc2*cos(q2), I2 + m2*l1*lc2*cos(q2);
                      I2 + m2*l1*lc2*cos(q2),                    I2];

C = [-2*m2*l1*lc2*sin(q2)*dq2, -m2*l1*lc2*sin(q2)*dq2;
        m2*l1*lc2*sin(q2)*dq1,                     0];
    
G = [-m1*g*lc1*cos(q1) - m2*g*(l1*cos(q1) + lc2*cos(q1+q2));
                                    -m2*g*lc2*cos(q1 + q2)];
 
V = C*[dq1;dq2];

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

Kp = 100*[1, 0; 0, 1];
Kv = 0.01*Kp;

tf1 = 4.9;

syms t_

y = (l1^2 + 2*cos(q1)*l1*l2 + l2^2)^(1/2);

vec_t = [-1.48353; 0.096];
% vec_t = [0.4364/5; 0.4364/5];
pos_d = sym(zeros(2,1));
pos_d(1,1) = -1.48353;
pos_d(2,1) = vec_t(2,1)*(sin(10*t_)-1);
% dvec_t = [1.4; 1.96];
dvec_t = [0; 0.96];
% dvec_t = [2.18/5; 2.18/5];
vel_d(1:2,1) = dvec_t*cos(10*t_);
ddvec_t = [-17.5/5; -10.9/5];
% ddvec_t = [-10.9/5; -10.9/5];
acc_d(1:2,1) = 2*ddvec_t*sin(10*t_);

q_init = double(subs(pos_d,t_,0))
dq_init = double(subs(vel_d,t_,0))
q_init = [-1.48353; -1.13446]
dq_init = [0; 0];
ddq_init = double(subs(acc_d,t_,0))
x_init = [q_init;dq_init];
q_d = double(subs(pos_d,t_,tf1))
dq_d = double(subs(vel_d,t_,tf1))
ddq_d = double(subs(acc_d,t_,tf1))
x_d = [q_d; dq_d];

trajectory_coefficients = [vec_t; dvec_t; ddvec_t];

L = 50*eye(12);
% [a0 + a1x + a2x^2 + a3x^3]
alpha0 = [0.1; -0.01; 0.001; -0.0001; 0.00001; -0.000001; 0.1; -0.01; 0.001; -0.0001; 0.00001; -0.000001];
% alpha0 = [0; 0; 0; 0; 0; 0; 0; 0; 0; 0; 0; 0; 0; 0];
% alpha0 = [0; 0; 0; 0; 0; 0; 0];

% [T1,X1] = ode15s(@(t,x)simArm(x, x_d, Kp, Kv, ddq_d, cart_des_1_traj, t, 1, [0 tf1]),[0 tf1],x_init);
options = odeset('RelTol',1e-4,'AbsTol',[1e-4, 1e-4, 1e-4, 1e-4]);
[T1,X1] = ode15s(@(t,x)simAdaptivePhantomArm(x, trajectory_coefficients, x_d, Kp, Kv, L, t, alpha0, ddq_init),[0 tf1],x_init,options);
alpha
T = T1;
X = X1;

u = tau.';
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

