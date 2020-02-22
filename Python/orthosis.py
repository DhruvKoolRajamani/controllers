import scipy.integrate as integrate
import numpy as np
from numpy import sin, cos, pi
from controller_extensions import *
import matplotlib.pyplot as plt

np.set_printoptions(precision=3, linewidth=200)


class Orthosis(object):
  """ Default class to subscribe and publish to orthosis robots """

  LINK_LENGTH_1 = 0.04297
  LINK_LENGTH_1 = 0.04689
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
    model='pd',
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
    self.count = 0.
    self.prev_time = 0.

    self.state0 = np.array(x0)
    self.state = self.state0

    self.full_state = {}

    # Join Limits
    # [MIN_POS, MAX_POS, MIN_VEL, MAX_VEL]
    self.JOINT_LIMITS_1 = np.array(
      [0.,
       55 * pi / 180,
       -(55 * pi / 180) * 1e5 / 2,
       (55*pi/180) * 1e5 / 2]
    )
    self.JOINT_LIMITS_2 = np.array(
      [0.,
       80 * pi / 180,
       -(80 * pi / 180) * 1e5 / 2,
       (80*pi/180) * 1e5 / 2]
    )

    return

  def _desired_trajectory(
    self,
    t=0.,
    coeffs=[0.,
            0.],
    traj_type='sin',
    frequency=10,
    stiff_traj=False
  ):
    """  """

    trajectory = np.zeros((4,))
    if not stiff_traj:
      if traj_type == 'sin':
        trajectory[:2] = coeffs * (sin(frequency * t) + 1)
        trajectory[2:] = 2 * frequency * coeffs * cos(frequency * t)
    else:
      if traj_type == 'sin':
        trajectory[:2] = coeffs[:2] * (sin(frequency * t) + 1)
        trajectory[2:] = 2 * coeffs[:2] * cos(frequency * t)

    return trajectory

  def _state_error(self, states=[0., 0., 0., 0.], trajectory=[0., 0., 0., 0.]):
    """ """

    # state_error = np.zeros((4,))
    if self.model == 'pd':
      state_error = trajectory - states[:4]
    elif self.model == 'adaptive':
      state_error = (states[:4] - trajectory)
    else:
      print("Not specified")

    return state_error

  def generate_system(self):

    i = 0.0
    count = 0
    scale = 1e3
    Kp = 1 * np.eye(2) * scale
    Kv = 1e-2 * Kp

    traj_coeffs = np.array(
      [
        (self.JOINT_LIMITS_1[1] - self.JOINT_LIMITS_1[0]) / 10,
        (self.JOINT_LIMITS_2[1] - self.JOINT_LIMITS_2[0]) / 10
      ]
    ).reshape(2,
              )
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

    traj = self._desired_trajectory(
      t=0,
      coeffs=traj_coeffs,
      traj_type='sin',
      frequency=10 * scale,
      stiff_traj=False
    )
    self.state = traj
    dx = self.state

    plt.ion()
    fig = plt.figure()
    while i <= self.tf:
      # u = -(1 * self.error[:2] + 0.01 * self.error[2:4])
      traj = self._desired_trajectory(
        t=i,
        coeffs=traj_coeffs,
        traj_type='sin',
        frequency=10 * scale,
        stiff_traj=False
      )

      err = self._state_error(dx, traj)
      u = (Kp.dot(err[:2].reshape(2, 1)) + Kv.dot(err[2:].reshape(2, 1)))
      prev_time, dx = self.step(u, self.timestep)

      U1.append(u[0][0])
      U2.append(u[1][0])
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
      prev_state = np.append(s, torque)  # prev state with torque
    else:
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

    dx[0] = wrap(dx[0], self.JOINT_LIMITS_1[0], self.JOINT_LIMITS_1[1])
    dx[1] = wrap(dx[1], self.JOINT_LIMITS_2[0], self.JOINT_LIMITS_2[1])
    dx[2] = bound(dx[2], self.JOINT_LIMITS_1[2], self.JOINT_LIMITS_1[3])
    dx[3] = bound(dx[3], self.JOINT_LIMITS_2[2], self.JOINT_LIMITS_2[3])

    self.prev_time = self.count
    self.count = 0.

    self.state = dx

    # Add loss function here if model is not rl
    # Option to override

    return self.prev_time, dx

  def _ode_func(self, t, states, inputs):
    """dx/dt"""
    u = inputs
    s = states
    y = None

    if self.input_shape == (1, 1):
      u = np.array([0., u]).reshape(2, 1)
    else:
      u = np.array(u).reshape(self.input_shape[0], 1)

    ds = np.zeros(s.shape)
    # ds = s
    ds[0] = s[2]
    ds[1] = s[3]
    mass_mat, cor_mat, grav_mat, stiffness = self.state_dynamics(s)
    ds[2:] = np.linalg.solve(mass_mat, u - stiffness).reshape(2,)

    if self.model is not 'rl':
      y = np.array([ds[0], ds[1], ds[2], ds[3]])

    return y

  def _dxdt(self, t, states):
    """dx/dt"""

    s = None
    y = None

    s = states[:4]
    u = states[4:]

    if self.input_shape == (1, 1):
      # Pad 0 to joint 1 if shape if 1,1
      u = np.array([0., u]).reshape(2, 1)
    else:
      u = np.array(u).reshape(self.input_shape[0], 1)

      # IF NON PD ADD FUNCTION HERE FOR FORWARD PASS

    ds = np.zeros(s.shape)
    ds[0] = s[2]
    ds[1] = s[3]
    mass_mat, cor_mat, grav_mat, stiffness = self.state_dynamics(s)
    ds[2:] = np.linalg.solve(mass_mat, u - stiffness).reshape(2,)

    self.count += t

    if self.input_shape == (1, 1):
      # Pad 0 to joint 1 if shape if 1,1
      y = np.array([ds[0], ds[1], ds[2], ds[3], u[1][0]])
    else:
      y = np.array([ds[0], ds[1], ds[2], ds[3], u[0][0], u[1][0]])

    return y

  def state_dynamics(self, s):
    mass_mat = np.array(
      [
        (
          1.38e-5 * cos(s[0]) - 6.04e-6 * cos(s[0])**2 +
          6.04e-6 * cos(s[0])**2 * cos(s[0])**2 -
          7.68e-6 * cos(s[0]) * sin(s[0]) * sin(s[0]) -
          8.38e-6 * cos(s[0]) * cos(s[0]) * sin(s[0]) * sin(s[0]) + 2.33e-5
        ),
        (
          3.04e-6 * cos(s[0]) - 4.19e-6 * cos(s[0])**2 -
          4.19e-6 * cos(s[0])**2 + 3.84e-6 * cos(s[0])**2 * cos(s[0]) +
          8.38e-6 * cos(s[0])**2 * cos(s[0])**2 -
          6.04e-6 * cos(s[0]) * cos(s[0]) * sin(s[0]) * sin(s[0]) + 7.63e-6
        ),
        (
          4.96e-6 * cos(s[0]) + 2.1e-6 * cos(2.0 * s[0]) *
          (2.0 * cos(s[0])**2 - 1.0) + 1.92e-6 * cos(2.0 * s[0]) * cos(s[0]) +
          2.0e-39 * sin(2.0 * s[0]) * sin(s[0]) -
          3.02e-6 * sin(2.0 * s[0]) * cos(s[0]) * sin(s[0]) + 5.53e-6
        ),
        (
          6.04e-6 * cos(s[0])**2 * cos(s[0])**2 - 6.04e-6 * cos(s[0])**2 -
          8.38e-6 * cos(s[0]) * cos(s[0]) * sin(s[0]) * sin(s[0]) + 7.63e-6
        )
      ]
    ).reshape((2,
               2))
    cor_mat = np.array(
      [
        (
          4.53e-6 * s[2]**2 * sin(2.0 * s[0]) +
          1.51e-6 * s[3]**2 * sin(2.0 * s[0]) -
          6.29e-6 * s[2]**2 * cos(2.0 * s[0]) * sin(2.0 * s[0]) -
          4.53e-6 * s[2]**2 * cos(2.0 * s[0]) * sin(2.0 * s[0]) +
          2.1e-6 * s[3]**2 * cos(2.0 * s[0]) * sin(2.0 * s[0]) +
          1.51e-6 * s[3]**2 * cos(2.0 * s[0]) * sin(2.0 * s[0]) -
          1.15e-5 * s[2]**2 * cos(2.0 * s[0]) * sin(s[0]) -
          3.02e-6 * s[2] * s[3] * cos(2.0 * s[0]) * sin(2.0 * s[0]) -
          4.19e-6 * s[2] * s[3] * cos(2.0 * s[0]) * sin(2.0 * s[0]) -
          3.84e-6 * s[2] * s[3] * sin(2.0 * s[0]) * cos(s[0])
        ),
        (
          6.88e-6 * s[2]**2 * sin(s[0]) + 1.51e-6 * s[2]**2 * sin(2.0 * s[0]) +
          4.53e-6 * s[3]**2 * sin(2.0 * s[0]) -
          4.96e-6 * s[2] * s[3] * sin(s[0]) +
          1.51e-6 * s[2]**2 * cos(2.0 * s[0]) * sin(2.0 * s[0]) +
          2.1e-6 * s[2]**2 * cos(2.0 * s[0]) * sin(2.0 * s[0]) -
          4.53e-6 * s[3]**2 * cos(2.0 * s[0]) * sin(2.0 * s[0]) -
          6.29e-6 * s[3]**2 * cos(2.0 * s[0]) * sin(2.0 * s[0]) +
          1.92e-6 * s[2]**2 * sin(2.0 * s[0]) * cos(s[0]) -
          4.19e-6 * s[2] * s[3] * cos(2.0 * s[0]) * sin(2.0 * s[0]) -
          3.02e-6 * s[2] * s[3] * cos(2.0 * s[0]) * sin(2.0 * s[0]) -
          1.92e-6 * s[2] * s[3] * cos(2.0 * s[0]) * sin(s[0])
        )
      ]
    ).reshape((2,
               1))
    cor_mat = np.array(
      [
        (
          4.53e-6 * s[2]**2 * sin(2.0 * s[0]) +
          1.51e-6 * s[3]**2 * sin(2.0 * s[0]) -
          6.29e-6 * s[2]**2 * cos(2.0 * s[0]) * sin(2.0 * s[0]) -
          4.53e-6 * s[2]**2 * cos(2.0 * s[0]) * sin(2.0 * s[0]) +
          2.1e-6 * s[3]**2 * cos(2.0 * s[0]) * sin(2.0 * s[0]) +
          1.51e-6 * s[3]**2 * cos(2.0 * s[0]) * sin(2.0 * s[0]) -
          1.15e-5 * s[2]**2 * cos(2.0 * s[0]) * sin(s[0]) -
          3.02e-6 * s[2] * s[3] * cos(2.0 * s[0]) * sin(2.0 * s[0]) -
          4.19e-6 * s[2] * s[3] * cos(2.0 * s[0]) * sin(2.0 * s[0]) -
          3.84e-6 * s[2] * s[3] * sin(2.0 * s[0]) * cos(s[0])
        ),
        (
          6.88e-6 * s[2]**2 * sin(s[0]) + 1.51e-6 * s[2]**2 * sin(2.0 * s[0]) +
          4.53e-6 * s[3]**2 * sin(2.0 * s[0]) -
          4.96e-6 * s[2] * s[3] * sin(s[0]) +
          1.51e-6 * s[2]**2 * cos(2.0 * s[0]) * sin(2.0 * s[0]) +
          2.1e-6 * s[2]**2 * cos(2.0 * s[0]) * sin(2.0 * s[0]) -
          4.53e-6 * s[3]**2 * cos(2.0 * s[0]) * sin(2.0 * s[0]) -
          6.29e-6 * s[3]**2 * cos(2.0 * s[0]) * sin(2.0 * s[0]) +
          1.92e-6 * s[2]**2 * sin(2.0 * s[0]) * cos(s[0]) -
          4.19e-6 * s[2] * s[3] * cos(2.0 * s[0]) * sin(2.0 * s[0]) -
          3.02e-6 * s[2] * s[3] * cos(2.0 * s[0]) * sin(2.0 * s[0]) -
          1.92e-6 * s[2] * s[3] * cos(2.0 * s[0]) * sin(s[0])
        )
      ]
    ).reshape((2,
               1))
    grav_mat = np.array(
      [
        (-0.00157 * cos(s[0] + s[0]) - 0.00352 * cos(s[0])),
        (-0.00157 * cos(s[0] + s[0]))
      ]
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
        (s[0] * (180/pi))**5,
        (s[0] * (180/pi))**4,
        (s[0] * (180/pi))**3,
        (s[0] * (180/pi))**2,
        (s[0] * (180/pi)),
        1,
        (s[1] * (180/pi))**5,
        (s[1] * (180/pi))**4,
        (s[1] * (180/pi))**3,
        (s[1] * (180/pi))**2,
        (s[1] * (180/pi)),
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
    AVAIL_TORQUE =\n
        [{}, {}, {}]\n,
    MASS_MAT =\n,
    {},\n
    CORIOLIS_MAT =\n,
    {},\n
    GRAVITY_MAT =\n,
    {}\n,
    STIFFNESS_RELAXED =\n
    {},\n
    STIFFNESS_EXTENDED =\n
    {},\n
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
      self.JOINT_LIMITS_2[3],
      self.AVAIL_TORQUE[0],
      self.AVAIL_TORQUE[1],
      self.AVAIL_TORQUE[2],
      self.mass_mat,
      self.cor_mat,
      self.grav_mat,
      self.stiffness_relaxed_coeffs.dot(self.stiffness_states),
      self.stiffness_extended_coeffs.dot(self.stiffness_states)
    )

    return objstring


# def adaptiveController():

#   return

if __name__ == '__main__':
  # Tests
  orthosis = Orthosis(input_shape=(2, 1), model='pd')
  orthosis.generate_system()