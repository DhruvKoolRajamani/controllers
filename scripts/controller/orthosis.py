import scipy.integrate as integrate
import numpy as np
from numpy import sin, cos, pi
from controller.controller_extensions import *
import matplotlib.pyplot as plt

np.set_printoptions(precision=3, linewidth=200)


class Orthosis(object):
  """ Default class to subscribe and publish to orthosis robots """

  LINK_LENGTH_1 = 1.
  LINK_LENGTH_2 = 1.
  LINK_MASS_1 = 0.0049366
  LINK_MASS_2 = 0.0034145

  def __init__(
    self,
    x0=[0.,
        0.,
        0.,
        0.],
    input_shape=(1,
                 1),
    timestep=1e-7,
    model='rl',
    tf=2
  ):
    """
    Initialize the class with the specific params
  *x0*
      initial state array of size 4 (len(x0) == 4)
  *input_shape*
      (1,1) -> single input
      (2,1) -> fully actuated
  *timestep*
      timestep to integrate over. This is replaced by self.dt in the case of RL
  *model*
      model options: ['rl', 'adaptive', 'ml', 'pd']
  *tf*
      final time if model is not rl
  """

    super(Orthosis, self).__init__()

    # Frequency of sinwave trajectory
    self.input_shape = input_shape
    self.timestep = timestep
    self.tf = tf
    self.model = model
    # Store previous time before each step call
    self.count = 0
    self.prev_time = 0.

    self.state0 = np.array(x0)
    self.state = self.state0

    self.full_state = {}
    self.exclusive_traj = None

    # Scaling factor for frequency
    self.scale = 1e3

    # Join Limits
    # [MIN_POS, MAX_POS, MIN_VEL, MAX_VEL]
    # self.JOINT_LIMITS_1 = np.array(
    #   [0.,
    #    55 * pi / 180,
    #    -(55 * pi / 180) * 1e5 / 2,
    #    (55*pi/180) * 1e5 / 2]
    # )
    # self.JOINT_LIMITS_2 = np.array(
    #   [0.,
    #    80 * pi / 180,
    #    -(80 * pi / 180) * 1e5 / 2,
    #    (80*pi/180) * 1e5 / 2]
    # )
    # self.JOINT_LIMITS_1 = np.array(
    #   [0.,
    #    55 * pi / 180,
    #    -1487.5 * self.scale,
    #    1487.5 * self.scale]
    # )
    # self.JOINT_LIMITS_2 = np.array(
    #   [0.,
    #    80 * pi / 180,
    #    -1487.5 * self.scale,
    #    1487.5 * self.scale]
    # )
    self.JOINT_LIMITS_1 = np.array(
      [-80 * pi / 180,
       0.,
       -1487.5 * self.scale,
       1487.5 * self.scale]
    )
    self.JOINT_LIMITS_2 = np.array(
      [-55 * pi / 180,
       0.,
       -1487.5 * self.scale,
       1487.5 * self.scale]
    )
    self.TAU_LIMITS_1 = np.array([-6.818 * self.scale, 6.818 * self.scale])
    self.TAU_LIMITS_2 = np.array([-6.818 * self.scale, 6.818 * self.scale])

    self.traj_coeffs = np.array(
      [
        (self.JOINT_LIMITS_1[1] - self.JOINT_LIMITS_1[0]) / 10,
        (self.JOINT_LIMITS_2[1] - self.JOINT_LIMITS_2[0]) / 10
      ]
    ).reshape(2,
              )

    return

  def _desired_trajectory(
    self,
    t=0.,
    coeffs=[0.,
            0.],
    traj_type='sin',
    frequency=10,
    stiff_traj=False,
    alpha=0,
    beta=0
  ):
    """  """

    trajectory = np.zeros((4,))
    if not stiff_traj:
      if traj_type == 'sin':
        trajectory[:2] = coeffs * (sin(frequency*t + alpha) - 1 - beta)
        trajectory[2:] = 2 * frequency * coeffs * cos(frequency*t + alpha)
    else:
      if traj_type == 'sin':
        trajectory[:2] = coeffs[:2] * (sin(frequency*t + alpha) - 1 - beta)
        trajectory[2:] = 2 * coeffs[:2] * cos(frequency * t)
    # print(coeffs, 2 * frequency * coeffs, alpha, beta)
    return trajectory

  def set_exclusive_traj(self, joint):
    """ Set joint for exclusive trajectory """

    j1_des = self.traj_coeffs[0]
    j2_des = self.traj_coeffs[1]

    if joint == 1:
      self.exclusive_traj = 1
      j2_des = 0.
    else:
      self.exclusive_traj = 2
      j1_des = 0.

    self.traj_coeffs = np.array([j1_des, j2_des])

    return

  def _state_error(self, states=[0., 0., 0., 0.], trajectory=[0., 0., 0., 0.]):
    """ """

    # state_error = np.zeros((4,))
    if self.model == 'pd':
      state_error = trajectory - states[:4]
      print(state_error)
    elif self.model == 'adaptive':
      state_error = (states[:4] - trajectory)
    else:
      state_error = (states[:4] - trajectory)

    return state_error

  def generate_system(self):

    i = 0.0
    count = 0
    Kp = 100 * np.eye(2) * self.scale
    Kv = 1e-2 * Kp

    # traj_coeffs = np.array(
    #   [
    #     (self.JOINT_LIMITS_1[1] - self.JOINT_LIMITS_1[0]) / 10,
    #     (self.JOINT_LIMITS_2[1] - self.JOINT_LIMITS_2[0]) / 10
    #   ]
    # ).reshape(2,
    #           )
    # traj_coeffs = np.array([0.6982 / 5, 0.4364 / 5, 3.49 / 5, 2.18 / 5])

    U1 = []
    U2 = []
    J1 = []
    J2 = []
    T1 = []
    T2 = []
    E1 = []
    E2 = []
    t = []
    T = []

    # self.set_exclusive_traj(1)

    traj = self._desired_trajectory(
      t=0,
      coeffs=self.traj_coeffs,
      traj_type='sin',
      frequency=10 * self.scale,
      stiff_traj=False
    )
    self.state = traj
    self.state[2:] = 0.
    dx = self.state

    plt.ion()
    fig = plt.figure()
    while i <= self.tf:
      # u = -(1 * self.error[:2] + 0.01 * self.error[2:4])
      traj = self._desired_trajectory(
        t=i,
        coeffs=self.traj_coeffs,
        traj_type='sin',
        frequency=10 * self.scale,
        stiff_traj=False
      )

      err = self._state_error(dx, traj)
      u = (Kp.dot(err[:2].reshape(2, 1)) + Kv.dot(err[2:].reshape(2, 1)))

      if self.input_shape == (1, 1):
        u = u[1, 0]

      prev_time, dx = self.step(u, self.timestep)

      # U1.append(u[0][0])
      # U2.append(u[1][0])
      J1.append(dx[0])
      J2.append(dx[1])
      t.append(prev_time)
      T1.append(traj[0])
      T2.append(traj[1])
      # T.append(traj)
      E1.append(err[0])
      E2.append(err[1])

      i += prev_time
      count += 1

      if count % 100 == 0:
        plt.plot(J1, 'r')
        plt.plot(J2, 'b')
        plt.plot(T1, 'r--')
        plt.plot(T2, 'b--')
        plt.show()
        plt.pause(1e-4)
        # Trajectory_Max = np.array(T)
        # print(np.max(Trajectory_Max, axis=0))

    plt.subplot(311)
    j1_handle = plt.plot(J1, 'r')
    j2_handle = plt.plot(J2, 'b')
    t1_handle = plt.plot(T1, 'r--')
    t2_handle = plt.plot(T2, 'b--')
    plt.subplot(312)
    u1_handle = plt.plot(U1, 'r')
    u2_handle = plt.plot(U2, 'b')
    plt.subplot(313)
    e1_handle = plt.plot(E1, 'r')
    e2_handle = plt.plot(E2, 'b')

    plt.show()

    return

  def step(self, torque, timestep=0.1):
    # dx/dt = Ax + Bu
    # dx1(i+1) = dx3(i)
    # dx2(i+1) = dx4(i)
    # dx3(i+1) = inv(M(i))*(0 - V(i) - G(i) - Stiff(i))
    # dx4(i+1) = inv(M(i))*(u(i) - V(i) - G(i) - Stiff(i))

    s = self.state  # previous state without torque
    prev_state = s

    dx = None

    if self.input_shape == (1, 1):
      torque = bound(torque, self.TAU_LIMITS_1[0], self.TAU_LIMITS_1[1])
      tau = np.array([0., torque])
      prev_state = np.append(s, tau)  # prev state with torque
    else:
      torque[0] = bound(torque[0], self.TAU_LIMITS_1[0], self.TAU_LIMITS_1[1])
      torque[1] = bound(torque[1], self.TAU_LIMITS_2[0], self.TAU_LIMITS_2[1])
      prev_state = np.append(
        s,
        np.array(torque).reshape(1,
                                 self.input_shape[0])
      )

    # if self.model is not 'rl':
    #   # r = integrate.ode(self._ode_func).set_integrator('dopri5', verbosity=1)
    #   # # 'dopri5', verbosity=1
    #   # r.set_f_params(torque.reshape(self.input_shape[0]))
    #   # r.set_initial_value(y=prev_state, t=0.0)
    #   # dx = r.integrate(r.t + timestep)
    #   # self.prev_time = r.t
    #   rk45 = integrate.RK23(self._dxdt, 0, prev_state, t_bound=timestep)
    #   rk45.step()
    #   self.prev_time = rk45.t
    #   dx = rk45.y
    #   dx = dx[:4]
    # else:
    dx = rk4(self._dxdt, [0, timestep], prev_state)
    dx = dx[-1]
    dx = dx[:4]

    # Consider changing to bound bound bound bound
    dx[0] = bound(dx[0], self.JOINT_LIMITS_1[0], self.JOINT_LIMITS_1[1])
    dx[1] = bound(dx[1], self.JOINT_LIMITS_2[0], self.JOINT_LIMITS_2[1])
    dx[2] = bound(dx[2], self.JOINT_LIMITS_1[2], self.JOINT_LIMITS_1[3])
    dx[3] = bound(dx[3], self.JOINT_LIMITS_2[2], self.JOINT_LIMITS_2[3])

    cnt = self.count
    self.count = 0.

    self.state = dx

    # Add loss function here if model is not rl
    # Option to override

    return cnt, dx

  # def _ode_func(self, t, states, inputs):
  #   """dx/dt"""
  #   u = inputs
  #   s = states
  #   y = None

  #   if self.input_shape == (1, 1):
  #     u = np.array([0., u]).reshape(2, 1)
  #   else:
  #     u = np.array(u).reshape(self.input_shape[0], 1)

  #   ds = np.zeros(s.shape)
  #   # ds = s
  #   ds[0] = s[2]
  #   ds[1] = s[3]
  #   mass_mat, cor_mat, grav_mat, stiffness = self.state_dynamics(s)
  #   ds[2:] = np.linalg.solve(mass_mat, u - stiffness).reshape(2,)

  #   if self.model is not 'rl':
  #     y = np.array([ds[0], ds[1], ds[2], ds[3]])

  #   return y

  def _dxdt(self, t, states):
    """dx/dt"""

    s = None
    y = None

    s = states[:4]
    u = states[4:]

    u = np.array(u).reshape(u.size, 1)

    # IF NON PD ADD FUNCTION HERE FOR FORWARD PASS

    ds = np.zeros(s.shape)
    # ds[0] = s[2]
    # ds[1] = s[3]
    # mass_mat, cor_mat, grav_mat, stiffness = self.state_dynamics(s)
    # ds[2:] = np.linalg.solve(mass_mat, u - stiffness).reshape(2,)

    ds = self.phantom_hand_state_dynamics(s, u)

    # self.exclusive_traj

    self.count += t

    # if self.input_shape == (1, 1):
    #   # Pad 0 to joint 1 if shape if 1,1
    #   y = np.array([ds[0], ds[1], ds[2], ds[3], 0., u[1][0]])
    # else:
    y = np.array([ds[0], ds[1], ds[2], ds[3], u[0][0], u[1][0]])

    return y

  def phantom_hand_state_dynamics(self, s, u):
    """ """

    q1 = s[0]
    q2 = s[1]
    dq1 = s[2]
    dq2 = s[3]

    m11 = 0.00015561 * cos(q2) + 0.00027294
    m12 = 0.000077804 * cos(q2) + 0.0001231
    m21 = 0.000077804 * cos(q2) + 0.0001231
    m22 = 0.0001231

    c1 = 0
    c2 = 0.000077804 * sin(q2) * dq1**2 - 0.000077804 * dq2 * sin(q2) * dq1

    g1 = 0.030274 * cos(q1) + 0.017732 * cos(q1) * cos(q2) - 0.017732 * sin(
      q1
    ) * sin(q2)
    g2 = 0.017732 * cos(q1) * cos(q2) - 0.017732 * sin(q1) * sin(q2)

    stiffness_relaxed_coeffs = np.array(
      [
        0,
        -1.4724e-14,
        7.2021e-11,
        -1.5778e-7,
        1.0758e-4,
        0.14548,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        3.6689e-9,
        -6.6466e-7,
        4.3944e-5,
        -1.267e-3,
        1.3425e-2,
        3.056e-2
      ]
    ).reshape(2,
              12)
    stiffness_extended_coeffs = np.array(
      [
        2.4553e-9,
        -7.4367e-7,
        8.1609e-5,
        -3.9209e-3,
        6.6136e-2,
        0.34106,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        -1.565e-9,
        3.4856e-7,
        -2.1952e-5,
        2.9395e-4,
        3.0018e-3,
        0.23896
      ]
    ).reshape(2,
              12)

    e1 = q1 - self.JOINT_LIMITS_1[0]
    e2 = q2 - self.JOINT_LIMITS_2[0]
    stiffness_states = np.array(
      [
        (e1 * (180/pi))**5,
        (e1 * (180/pi))**4,
        (e1 * (180/pi))**3,
        (e1 * (180/pi))**2,
        (e1 * (180/pi)),
        1,
        (e2 * (180/pi))**5,
        (e2 * (180/pi))**4,
        (e2 * (180/pi))**3,
        (e2 * (180/pi))**2,
        (e2 * (180/pi)),
        1
      ]
    ).reshape(12,
              1)

    mass_mat = np.append([[m11, m12]], [[m21, m22]]).reshape(2, 2)
    cor_mat = np.append([[c1]], [[c2]]).reshape(2, 1)
    stiffness = stiffness_relaxed_coeffs.dot(stiffness_states)

    dx1 = dq1
    dx2 = dq2
    # yapf: disable
    dx4 = (1 / (m22 - (m21/m11) * m12)) * (
      u[1,0] - (c2 + g2 + stiffness[1,0]) + (m21/m11) * (
        c1 + g1 + stiffness[0,0]
      )
    )
    dx3 = -(1 / m11) * (m12*dx4 + (c1 + g1 - stiffness[0, 0]))
    dx = np.array([dx1, dx2, dx3, dx4])
    # yapf: enable

    return dx

  def simple_state_dynamics(self, s, u):

    x1 = s[0]
    x2 = s[1]
    x3 = s[2]
    x4 = s[3]

    dx1 = 0.
    dx2 = 0.
    dx3 = 0.
    dx4 = 0.

    # Testing stiffness only
    # u = np.array([0, 0]).reshape(2, 1)

    # if self.exclusive_traj == 2:
    #   # Joint 1 is locked
    #   x3 = 0.
    #   dx3 = 0.

    # Mass Matrix
    m11 = 1.38e-5 * cos(x2) - 6.04e-6 * cos(x1)**2 + 6.04e-6 * cos(
      x1
    )**2 * cos(x2)**2 - 7.68e-6 * cos(x1) * sin(x1) * sin(x2) - 8.38e-6 * cos(
      x1
    ) * cos(x2) * sin(x1) * sin(x2) + 2.33e-5
    m12 = 3.04e-6 * cos(x2) - 4.19e-6 * cos(x1)**2 - 4.19e-6 * cos(
      x2
    )**2 + 3.84e-6 * cos(x1)**2 * cos(x2) + 8.38e-6 * cos(x1)**2 * cos(
      x2
    )**2 - 6.04e-6 * cos(x1) * cos(x2) * sin(x1) * sin(x2) + 7.63e-6
    m21 = 4.96e-6 * cos(x2) + 2.1e-6 * cos(2.0 * x1) * (
      2.0 * cos(x2)**2 - 1.0
    ) + 1.92e-6 * cos(2.0 * x1) * cos(x2) + 2.0e-39 * sin(
      2.0 * x1
    ) * sin(x2) - 3.02e-6 * sin(2.0 * x1) * cos(x2) * sin(x2) + 5.53e-6
    m22 = 6.04e-6 * cos(x1)**2 * cos(x2)**2 - 6.04e-6 * cos(
      x2
    )**2 - 8.38e-6 * cos(x1) * cos(x2) * sin(x1) * sin(x2) + 7.63e-6

    # Stiffness + Gravity
    c11 = 0.00157 * sin(x1) * sin(x2) - 0.00157 * cos(x1) * cos(
      x2
    ) - 0.00352 * cos(x1) + 4.53e-6 * x3**2 * sin(
      2.0 * x1
    ) + 1.51e-6 * x4**2 * sin(2.0 * x1) - 6.29e-6 * x3**2 * cos(2.0 * x1) * sin(
      2.0 * x2
    ) - 4.53e-6 * x3**2 * cos(2.0 * x2) * sin(2.0 * x1) + 2.1e-6 * x4**2 * cos(
      2.0 * x1
    ) * sin(2.0 * x2) + 1.51e-6 * x4**2 * cos(2.0 * x2) * sin(
      2.0 * x1
    ) - 1.15e-5 * x3**2 * cos(2.0 * x1) * sin(x2) - 3.02e-6 * x3 * x4 * cos(
      2.0 * x1
    ) * sin(2.0 * x2) - 4.19e-6 * x3 * x4 * cos(2.0 * x2) * sin(
      2.0 * x1
    ) - 3.84e-6 * x3 * x4 * sin(2.0 * x1) * cos(x2)
    c21 = 6.88e-6 * x3**2 * sin(x2) - 0.00157 * cos(
      x1
    ) * cos(x2) + 0.00157 * sin(x1) * sin(x2) + 1.51e-6 * x3**2 * sin(
      2.0 * x2
    ) + 4.53e-6 * x4**2 * sin(
      2.0 * x2
    ) - 4.96e-6 * x3 * x4 * sin(x2) + 1.51e-6 * x3**2 * cos(2.0 * x1) * sin(
      2.0 * x2
    ) + 2.1e-6 * x3**2 * cos(2.0 * x2) * sin(2.0 * x1) - 4.53e-6 * x4**2 * cos(
      2.0 * x1
    ) * sin(2.0 * x2) - 6.29e-6 * x4**2 * cos(2.0 * x2) * sin(
      2.0 * x1
    ) + 1.92e-6 * x3**2 * sin(2.0 * x1) * cos(x2) - 4.19e-6 * x3 * x4 * cos(
      2.0 * x1
    ) * sin(2.0 * x2) - (3.02e-6 * x3 * x4 * cos(2.0 * x2) * sin(2.0 * x1)
                         ) - (1.92e-6 * x3 * x4 * cos(2.0 * x1) * sin(x2))

    grav_mat = np.array(
      [-0.00157 * cos(x1 + x2) - 0.00352 * cos(x1),
       -0.00157 * cos(x1 + x2)]
    ).reshape((2,
               1))

    stiffness_relaxed_coeffs = np.array(
      [
        0,
        -1.4724e-14,
        7.2021e-11,
        -1.5778e-7,
        1.0758e-4,
        0.14548,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        3.6689e-9,
        -6.6466e-7,
        4.3944e-5,
        -1.267e-3,
        1.3425e-2,
        3.056e-2
      ]
    ).reshape(2,
              12)
    stiffness_extended_coeffs = np.array(
      [
        2.4553e-9,
        -7.4367e-7,
        8.1609e-5,
        -3.9209e-3,
        6.6136e-2,
        0.34106,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        -1.565e-9,
        3.4856e-7,
        -2.1952e-5,
        2.9395e-4,
        3.0018e-3,
        0.23896
      ]
    ).reshape(2,
              12)

    e1 = self.JOINT_LIMITS_1[0] - x1
    e2 = self.JOINT_LIMITS_2[0] - x2
    stiffness_states = np.array(
      [
        (e1 * (180/pi))**5,
        (e1 * (180/pi))**4,
        (e1 * (180/pi))**3,
        (e1 * (180/pi))**2,
        (e1 * (180/pi)),
        1,
        (e2 * (180/pi))**5,
        (e2 * (180/pi))**4,
        (e2 * (180/pi))**3,
        (e2 * (180/pi))**2,
        (e2 * (180/pi)),
        1
      ]
    ).reshape(12,
              1)
    mass_mat = np.append([[m11, m12]], [[m21, m22]]).reshape(2, 2)
    cor_mat = np.append([[c11]], [[c21]]).reshape(2, 1)
    stiffness = stiffness_relaxed_coeffs.dot(stiffness_states)

    dx = None

    # if self.exclusive_traj == 1:
    #   # Joint 1 is locked
    #   dx1 = x3
    #   dx2 = x4
    #   dx3 = (
    #     u[1,
    #       0] - (-c21 - stiffness[1,
    #                              0]) - ((-c11 - stiffness[0,
    #                                                       0]) / m12)
    #   ) / (
    #     m21 - (m22*m11) / m12
    #   )
    #   dx4 = u[1, 0]
    #   dx = np.array([dx1, dx2, dx3, dx4])
    # elif self.exclusive_traj == 2:
    #   dx1 = x3
    #   dx2 = x4
    #   dx3 = 0.
    #   dx4 = u[1, 0] / m22 - (-c21 - stiffness[1, 0]) / m22  #grav_mat[1, 0] -
    #   dx = np.array([dx1, dx2, dx3, dx4])
    # else:
    # dx = np.linalg.solve(mass_mat, u - stiffness).reshape(2,)
    dx1 = x3
    dx2 = x4
    dx4 = (1 / (m22 - (m21/m11) * m12)) * (
      u[1,
        0] - (-c21 - grav_mat[1,
                              0] - stiffness[1,
                                             0]) + (m21/m11) *
      (-c11 - grav_mat[0,
                       0] - stiffness[0,
                                      0])
    )
    dx3 = -(1 / m11) * (m12*dx4 + (-c11 - grav_mat[0, 0] - stiffness[0, 0]))
    dx = np.array([dx1, dx2, dx3, dx4])

    return dx

  # Consider changing dynamics
  def state_dynamics(self, s):

    x1 = s[0]
    x2 = s[1]
    x3 = s[2]
    x4 = s[3]

    mass_mat = np.array(
      [
        (
          1.38e-5 * cos(x2) - 6.04e-6 * cos(x1)**2 +
          6.04e-6 * cos(x1)**2 * cos(x2)**2 -
          7.68e-6 * cos(x1) * sin(x1) * sin(x2) -
          8.38e-6 * cos(x1) * cos(x2) * sin(x1) * sin(x2) + 2.33e-5
        ),
        (
          3.04e-6 * cos(x2) - 4.19e-6 * cos(x1)**2 - 4.19e-6 * cos(x2)**2 +
          3.84e-6 * cos(x1)**2 * cos(x2) + 8.38e-6 * cos(x1)**2 * cos(x2)**2 -
          6.04e-6 * cos(x1) * cos(x2) * sin(x1) * sin(x2) + 7.63e-6
        ),
        (
          4.96e-6 * cos(x2) + 2.1e-6 * cos(2.0 * x1) *
          (2.0 * cos(x2)**2 - 1.0) + 1.92e-6 * cos(2.0 * x1) * cos(x2) +
          2.0e-39 * sin(2.0 * x1) * sin(x2) -
          3.02e-6 * sin(2.0 * x1) * cos(x2) * sin(x2) + 5.53e-6
        ),
        (
          6.04e-6 * cos(x1)**2 * cos(x2)**2 - 6.04e-6 * cos(x2)**2 -
          8.38e-6 * cos(x1) * cos(x2) * sin(x1) * sin(x2) + 7.63e-6
        )
      ]
    ).reshape((2,
               2))
    cor_mat = np.array(
      [
        (
          4.53e-6 * x3**2 * sin(2.0 * x1) + 1.51e-6 * x4**2 * sin(2.0 * x1) -
          6.29e-6 * x3**2 * cos(2.0 * x1) * sin(2.0 * x2) -
          4.53e-6 * x3**2 * cos(2.0 * x2) * sin(2.0 * x1) +
          2.1e-6 * x4**2 * cos(2.0 * x1) * sin(2.0 * x2) +
          1.51e-6 * x4**2 * cos(2.0 * x2) * sin(2.0 * x1) -
          1.15e-5 * x3**2 * cos(2.0 * x1) * sin(x2) -
          3.02e-6 * x3 * x4 * cos(2.0 * x1) * sin(2.0 * x2) -
          4.19e-6 * x3 * x4 * cos(2.0 * x2) * sin(2.0 * x1) -
          3.84e-6 * x3 * x4 * sin(2.0 * x1) * cos(x2)
        ),
        (
          6.88e-6 * x3**2 * sin(x2) + 1.51e-6 * x3**2 * sin(2.0 * x2) +
          4.53e-6 * x4**2 * sin(2.0 * x2) - 4.96e-6 * x3 * x4 * sin(x2) +
          1.51e-6 * x3**2 * cos(2.0 * x1) * sin(2.0 * x2) +
          2.1e-6 * x3**2 * cos(2.0 * x2) * sin(2.0 * x1) -
          4.53e-6 * x4**2 * cos(2.0 * x1) * sin(2.0 * x2) -
          6.29e-6 * x4**2 * cos(2.0 * x2) * sin(2.0 * x1) +
          1.92e-6 * x3**2 * sin(2.0 * x1) * cos(x2) -
          4.19e-6 * x3 * x4 * cos(2.0 * x1) * sin(2.0 * x2) -
          3.02e-6 * x3 * x4 * cos(2.0 * x2) * sin(2.0 * x1) -
          1.92e-6 * x3 * x4 * cos(2.0 * x1) * sin(x2)
        )
      ]
    ).reshape((2,
               1))
    grav_mat = np.array(
      [-0.00157 * cos(x1 + x2) - 0.00352 * cos(x1),
       -0.00157 * cos(x1 + x2)]
    ).reshape((2,
               1))

    stiffness_relaxed_coeffs = np.array(
      [
        0,
        -1.4724e-14,
        7.2021e-11,
        -1.5778e-7,
        1.0758e-4,
        0.14548,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        3.6689e-9,
        -6.6466e-7,
        4.3944e-5,
        -1.267e-3,
        1.3425e-2,
        3.056e-2
      ]
    ).reshape(2,
              12)
    stiffness_extended_coeffs = np.array(
      [
        2.4553e-9,
        -7.4367e-7,
        8.1609e-5,
        -3.9209e-3,
        6.6136e-2,
        0.34106,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        -1.565e-9,
        3.4856e-7,
        -2.1952e-5,
        2.9395e-4,
        3.0018e-3,
        0.23896
      ]
    ).reshape(2,
              12)
    stiffness_states = np.array(
      [
        (x1 * (180/pi))**5,
        (x1 * (180/pi))**4,
        (x1 * (180/pi))**3,
        (x1 * (180/pi))**2,
        (x1 * (180/pi)),
        1,
        (x2 * (180/pi))**5,
        (x2 * (180/pi))**4,
        (x2 * (180/pi))**3,
        (x2 * (180/pi))**2,
        (x2 * (180/pi)),
        1
      ]
    ).reshape(12,
              1)

    stiffness = stiffness_relaxed_coeffs.dot(stiffness_states)

    return mass_mat, cor_mat, grav_mat, stiffness

  def toString(self):
    objstring = """
    Constant Parameters:\n
    LINK_LENGTH_1 = {}\n
    LINK_LENGTH_1 = {}\n
    LINK_MASS_1 = {}\n
    LINK_MASS_2 = {}\n

    Variable Parameters:\n
    JOINT_LIMITS_1 =\n
        MIN_POS = {},\n
        MAX_POS = {},\n
        MIN_VEL = {},\n
        MAX_VEL = {}\n
    JOINT_LIMITS_2 =\n
        MIN_POS = {},\n
        MAX_POS = {},\n
        MIN_VEL = {},\n
        MAX_VEL = {},\n
    """.format(
      self.LINK_LENGTH_1,
      self.LINK_LENGTH_1,
      self.LINK_MASS_1,
      self.LINK_MASS_2,
      self.JOINT_LIMITS_1[0],
      self.JOINT_LIMITS_1[1],
      self.JOINT_LIMITS_1[2],
      self.JOINT_LIMITS_1[3],
      self.JOINT_LIMITS_2[0],
      self.JOINT_LIMITS_2[1],
      self.JOINT_LIMITS_2[2],
      self.JOINT_LIMITS_2[3]
    )

    return objstring


# def adaptiveController():

#   return

if __name__ == '__main__':
  # Tests
  orthosis = Orthosis(input_shape=(1, 1), model='pd')
  orthosis.generate_system()