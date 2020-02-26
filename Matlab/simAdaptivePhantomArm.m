function [dx] = simAdaptivePhantomArm(x, t_coeffs, x_d, Kp, Kv, L, t, alpha0, ddx_init)
%simAdaptiveArm Summary of this function goes here
%   Detailed explanation goes here
global p
global tm1
global tau
global err
global alpha
global act_tau
global zeta
global s
global counter
persistent k
persistent ddx

if t == 0
    ddx(1:2,1) = [0; 0];
end


l2 = 0.05604;
l1 = 0.043;

J = [- 0.043*sin(x(1,1)) - 0.05604*cos(x(1,1))*sin(x(2,1)) - 0.05604*cos(x(2,1))*sin(x(1,1)), - 0.05604*cos(x(1,1))*sin(x(2,1)) - 0.05604*cos(x(2,1))*sin(x(1,1));
     0.043*cos(x(1,1)) + 0.05604*cos(x(1,1))*cos(x(2,1)) - 0.05604*sin(x(1,1))*sin(x(2,1)),   0.05604*cos(x(1,1))*cos(x(2,1)) - 0.05604*sin(x(1,1))*sin(x(2,1))];

tm1 = [tm1; t];

M = [ 0.00021007*cos(x(2,1)) + 0.00026447, 0.00010504*sin(x(2,1)) + 0.00014401;
      0.00010504*sin(x(2,1)) + 0.00014401,                         0.00014401];

V = [  - 0.00021007*x(3,1)*x(4,1)*sin(x(2,1)) - 0.00010504*x(4,1)^2*sin(x(2,1));
                                               0.00010504*x(3,1)^2*sin(x(2,1))];

G = [ - 0.030316*cos(x(1,1)) - 0.023939*cos(x(1,1) + x(2,1));
                             -0.023939*cos(x(1,1) + x(2,1))];
                         
Tg = -G;


% Relaxed:
% 1: 5.0245e-10*x^5 - 1.4929e-7*x^4 + 0.000015689*x^3 - 0.00066357*x^2 + 0.0058206*x + 0.31472
% 1: 1.1562e-9*x^5 - 2.9833e-7*x^4 + 0.000026864*x^3 - 0.00097147*x^2 + 0.011926*x + 0.27389
% 
% 2: - 1.0703e-9*x^5 + 2.4455e-7*x^4 - 0.000022696*x^3 + 0.0010298*x^2 - 0.024215*x + 0.37904
% 2: - 3.1658e-9*x^5 + 7.0997e-7*x^4 - 0.000053963*x^3 + 0.0015456*x^2 - 0.011943*x + 0.085096
% 
% 3:
% Tau_Stiffness_Relaxed = [1.118e-18*(x(1,1)*(180/pi))^5 - 1.4724e-14*(x(1,1)*(180/pi))^4 + 7.2021e-11*(x(1,1)*(180/pi))^3 - 1.5778e-7*(x(1,1)*(180/pi))^2 + 0.00010758*(x(1,1)*(180/pi)) + 0.14548;
%                  3.6689e-9*(x(2,1)*(180/pi))^5 - 6.6466e-7*(x(2,1)*(180/pi))^4 + 0.000043944*(x(2,1)*(180/pi))^3 - 0.001267*(x(2,1)*(180/pi))^2 + 0.013425*(x(2,1)*(180/pi)) + 0.03056];
% 
% Tau_Stiffness_Extended = [2.4553e-9*(x(1,1)*(180/pi))^5 - 7.4367e-7*(x(1,1)*(180/pi))^4 + 0.000081609*(x(1,1)*(180/pi))^3 - 0.0039209*(x(1,1)*(180/pi))^2 + 0.066136*(x(1,1)*(180/pi)) + 0.34106;
%                           -1.565e-9*(x(2,1)*(180/pi))^5 + 3.4856e-7*(x(2,1)*(180/pi))^4 - 0.000021952*(x(2,1)*(180/pi))^3 + 0.00029395*(x(2,1)*(180/pi))^2 + 0.0030018*(x(2,1)*(180/pi)) + 0.23896]

Tau_Stiffness_Relaxed_coeffs = [0, -1.4724e-14, 7.2021e-11, - 1.5778e-7, 1.0758e-4, 0.14548, 0, 0, 0, 0, 0, 0;
                                0, 0, 0, 0, 0, 0, 3.6689e-9, -6.6466e-7, 4.3944e-5, -1.267e-3, 1.3425e-2, 3.056e-2];
Tau_Stiffness_Extended_coeffs = [2.4553e-9, -7.4367e-7, 8.1609e-5, -3.9209e-3, 6.6136e-2, 0.34106, 0, 0, 0, 0, 0, 0;
                                 0, 0, 0, 0, 0, 0, -1.565e-9, 3.4856e-7, -2.1952e-5, 2.9395e-4, 3.0018e-3, 0.23896];             
func_q = [((x(1,1) + 1.48353)*(180/pi))^5, ((x(1,1) + 1.48353)*(180/pi))^4, ((x(1,1) + 1.48353)*(180/pi))^3, ((x(1,1) + 1.48353)*(180/pi))^2, ((x(1,1) + 1.48353)*(180/pi)), 1, ((x(2,1)+1.13446)*(180/pi))^5, ((x(2,1)+1.13446)*(180/pi))^4, ((x(2,1)+1.13446)*(180/pi))^3, ((x(2,1)+1.13446)*(180/pi))^2, ((x(2,1)+1.13446)*(180/pi)), 1];
      
Tau_Stiffness_Relaxed = Tau_Stiffness_Relaxed_coeffs*func_q.';

Tau_Stiffness_Extended = Tau_Stiffness_Extended_coeffs*func_q.';

% Tau_Stiffness_Linear = [-(0.17236358338440383/85.6)*x(1,1)*(180/pi) + (15.690776837895514/85.6);
%                         -(0.07290796843590319/64.3)*x(2,1)*(180/pi) + (4.890171545518083/64.3)];
Tau_Stiffness = Tau_Stiffness_Relaxed;
% Tau_Stiffness = Tau_Stiffness_Linear;

if t >=5
    Tau_Stiffness = Tau_Stiffness_Extended;
end

% M_(ddx) + V_ + G_ + s_Tau(x) = u = Y*alpha_
% Initial Estimates
M_ = 0.00015*ones(2,2);
V_ = 0.000015*ones(2,1);
s_ = 0.1;

% Desired Trajectory
vec_t = t_coeffs(1:2,1);
x_d(1,1) = vec_t(1,1);
x_d(2,1) = vec_t(2,1)*(sin(10*t)-1);
dvec_t = t_coeffs(3:4,1);
x_d(3:4,1) = dvec_t*cos(10*t);
ddvec_t = t_coeffs(5:6,1);

e = x(1:2,1) - x_d(1:2,1);
err = [err e];
de = x(3:4,1) - x_d(3:4,1);

T = Tg - V - Tau_Stiffness;
T1 = T(1,1);
T2 = T(2,1);

if p == 1
    E = E + e*step;
    u = Kp*e + Kv*de + Ki*E;
elseif p == 2
    u = M*(ddq_d + Kp*e + Kv*de) + C*x(3:4,1);
elseif p == 3
    u = -(Kp*e + Kv*de);
elseif p == 4
    lambda = 2*Kp;
    s = de + lambda*e;
    % Adaptive control law
    x_unk_1 = [1 ((x(1,1) + 1.48353)*(180/pi)) ((x(1,1) + 1.48353)*(180/pi))^2 ((x(1,1) + 1.48353)*(180/pi))^3 ((x(1,1) + 1.48353)*(180/pi))^4 ((x(1,1) + 1.48353)*(180/pi))^5]; % de(1,1)
    x_unk_2 = [1 ((x(2,1)+1.13446)*(180/pi)) ((x(2,1)+1.13446)*(180/pi))^2 ((x(2,1)+1.13446)*(180/pi))^3 ((x(2,1)+1.13446)*(180/pi))^4 ((x(2,1)+1.13446)*(180/pi))^5]; % de(2,1)

    Y = [normpdf(x_unk_1) 0 0 0 0 0 0;
         0 0 0 0 0 0 normpdf(x_unk_2)];
    alpha_gradient = -L\Y.'*Kp*s; %
    if t == 0
        alpha = alpha0;
        k=0;
    else
        k = t-k;
        alpha = alpha + alpha_gradient*k;
    %     disp(alpha)
    end
    u = Y*alpha- Kp*s; %  %-

elseif p == 5
    if t == 0
        zeta = [0;0];
    end
    % NON Colloc Variables (k)
    % zeta_tilda = zeta - dx_d
    % s = dq - zeta
    lambda = Kp(1,1);
    Kn = 1;
    e_n = x(1,1) - x_d(1,1);
    de_n = x(3,1) - x_d(3,1);
%     s_n = de_n + lambda*e_n;
    s_n = x(3,1) - zeta(1);
    
    lambda = Kp(2,2);
    e_c = x(2,1) - x_d(2,1);
    de_c = x(4,1) - x_d(4,1);
%     s_c = de_c + lambda*e_c;
    s_c = x(4,1) - zeta(2);
    e = [e_n;e_c];
    err = [err e];
    s = [s_n;s_c];
    
    % Adaptive control law
   % Adaptive control law
    x_unk_1 = [1 x(1,1) x(1,1)^2 x(1,1)^3 x(1,1)^4 x(1,1)^5]; % de(1,1)
    x_unk_2 = [1 x(2,1) x(2,1)^2 x(2,1)^3 x(2,1)^4 x(2,1)^5]; % de(2,1)

    Y_n = [normpdf(x_unk_1) 0 0 0 0 0 0];
    Y_c =  [0 0 0 0 0 0 normpdf(x_unk_2)];
    Y = [Y_n;Y_c];
    
    
    S = [1 0];
    M_n = S*M_*S.';
    lambda1 = 100;
    lambda2 = 100;
    K=0.1;
%     L=0.1*eye(12);
    
    if t == 0
        k=0;
        alpha = alpha0;
        u = [0;Y_c*alpha - K*s_c]; %-
        zeta_gradient_n = M_n\(Kp(1,1)*s_n - Y_n*alpha);
        zeta_gradient_c = -lambda1*x(2,1) -lambda2*x(4,1);
        zeta_gradient = [zeta_gradient_n;zeta_gradient_c];
        zeta = zeta + zeta_gradient*t;
        
    else
        k = t-k;
        u = [0;Y_c*alpha - K*s_c]; %-
        alpha_gradient = -L\(Y.'*s); %
        alpha = alpha + alpha_gradient*k;
        zeta_gradient_n = M_n\(Kp(1,1)*s_n - Y_n*alpha);
        zeta_gradient_c = -lambda1*x(2,1) -lambda2*x(4,1);
        zeta_gradient = [zeta_gradient_n;zeta_gradient_c];
        zeta = zeta + zeta_gradient*t;
        
    %     disp(alpha)
    end

elseif p == 6
%     PFL Task Space

    y = (l1^2 + 2*cos(x(2,1))*l1*l2 + l2^2)^(1/2);
    % ddy_d = -(l1*l2*cos(x(2,1))*x(4,1)^2)/(l1^2 + 2*cos(x(2,1))*l1*l2 + l2^2)^(1/2) - (l1*l2*sin(x(2,1))*ddx(2,1))/(l1^2 + 2*cos(x(2,1))*l1*l2 + l2^2)^(1/2) - (l1^2*l2^2*sin(x(2,1))^2*x(4,1)^2)/(l1^2 + 2*cos(x(2,1))*l1*l2 + l2^2)^(3/2);
    ddy_d = ddx;
   
    dH = [0, 0];
    
    H1 = 0;
    H2 = -(l1*l2*sin(x(2,1)))/(l1^2 + 2*cos(x(2,1))*l1*l2 + l2^2)^(1/2);
    
    H_tilda = H2-(H1/M(1,1))*M(1,2);
    H_tilda_inv = pinv(H_tilda);
    
%     ddq2 = H_tilda_inv*(ddy_d - dH*x(3:4,1) - (H1/M(1,1))*T1);
    ddq2 = ddy_d(2,1);
    ddq1 = (1/M(1,1))*(T1 - M(1,2)*ddq2);
    
    u(1,1) = 0;
    u(2,1) = (M(2,1)/M(1,1))*(T1 - M(1,2)*ddy_d(2,1)) + M(2,2)*ddy_d(2,1) + T2;
%     u = [ddq1; ddq2]
    
else
    u = 0;
end

tau = [tau u];

dx(1,1) = x(3,1);
dx(2,1) = x(4,1);

if x(1,1) > 0 || x(1,1) < -1.48353
    dx(1,1) = 0;
end
if x(2,1) > 0 || x(2,1) < -1.13446
    dx(2,1) = 0;
end

dx(4,1) = (1/(M(2,2) - (M(2,1)/M(1,1))*M(1,2)))*(u(2,1) + T2 - (M(2,1)/M(1,1))*T1);
dx(3,1) = (1/M(1,1))*(T1 - M(1,2)*dx(4,1));


ddx = dx(3:4,1);
% dx(4,1) = (1/(M(2,2) - (M(2,1)/M(1,1))*M(1,2)))*(0 + T2 + (M(2,1)/M(1,1))*T1); %u(2,1)
% dx(3,1) = (1/M(1,1))*(T1 - M(1,2)*dx(4,1));
% dx(3:4,1) = M\([0; 0] - [T1; T2]); % u(2,1)

% dx(3:4,1) = M\(u + Tg - V -Tau_Stiffness); % u - V -  - Tau_Stiffness
k=t;
end
