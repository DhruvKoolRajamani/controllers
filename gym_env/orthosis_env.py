import scipy.integrate as integrate
import numpy as np
from numpy import sin, cos, pi
from orthosis_env import controller_extensions
import matplotlib.pyplot as plt
from orthosis_env.orthosis import Orthosis

from gym import core, spaces
from gym.utils import seeding


class RLController(Orthosis, core.Env):
  """ Adaptive Controller inherited from Orthosis """

  metadata = {
    'render.modes': ['human',
                     'rgb_array'],
    'video.frames_per_second': 15
  }

  dt = 0.000001
  AVAIL_TORQUE = [-0.01, 0., +0.01]

  torque_noise_max = 0.

  action_arrow = None
  domain_fig = None
  actions_num = 3

  def __init__(self, x0=[0., 0., 0., 0.], input_shape=(1, 1)):
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
      *alpha0*
          Initial state matrix for regressor coeffs
    """

    super(RLController,
          self).__init__(
            x0=x0,
            input_shape=input_shape,
            model='pd',
            timestep=self.dt
          )
    self.viewer = None

    high = self.JOINT_LIMITS_2
    high[0] = self.JOINT_LIMITS_1[1]
    high[2] = self.JOINT_LIMITS_1[3]

    low = self.JOINT_LIMITS_1
    low[1] = self.JOINT_LIMITS_2[0]
    low[3] = self.JOINT_LIMITS_2[2]

    # store at least 50 past values of trajectories and states
    self.past_trajectories = {}
    self.past_states = {}
    self.past_error = {}
    self.past_count = 0

    self.observation_space = spaces.Box(low=low, high=high, dtype=np.float32)
    self.action_space = spaces.Discrete(3)
    self.state = None
    self.traj_coeffs = np.array(
      [
        (self.JOINT_LIMITS_1[1] - self.JOINT_LIMITS_1[0]) / 10,
        (self.JOINT_LIMITS_2[1] - self.JOINT_LIMITS_2[0]) / 10
      ]
    ).reshape(2,
              )
    self.seed()

  def seed(self, seed=None):
    self.np_random, seed = seeding.np_random(seed)
    return [seed]

  def reset(self):
    self.state = self.np_random.uniform(
      low=0.1,
      high=0.3 * self.JOINT_LIMITS_1[1],
      size=(4,
            )
    )
    return self._get_ob()

  def past_state_pushback(self, traj, state):
    """ """

    if self.past_count < 50:
      self.past_states[i] = state
      self.past_trajectories[i] = traj
      self.past_error[i] = traj - state
      self.past_count += 1
    else:
      self.past_states.pop(0)
      self.past_states[50] = state
      self.past_trajectories.pop(0)
      self.past_trajectories[50] = traj
      self.past_error.pop(0)
      self.past_error[50] = traj - state

    return

  def _state_error(self, traj, state):

    max_joint_error = np.zeros((2,))
    err = 0.
    max_joint_error[0] = self.JOINT_LIMITS_1[1] - self.JOINT_LIMITS_1[0]
    max_joint_error[1] = self.JOINT_LIMITS_2[1] - self.JOINT_LIMITS_2[0]

    error = traj - state

    err = error[0] / max_joint_error[0] + error[1] / max_joint_error[1]

    return err

  def env_step(self, a, timestep=dt):

    traj = self._desired_trajectory(
      t=self.prev_time,
      coeffs=traj_coeffs,
      traj_type='sin',
      frequency=10,
      stiff_traj=False
    )

    torque = self.AVAIL_TORQUE[a]

    if self.torque_noise_max > 0:
      torque += self.np_random.uniform(
        -self.torque_noise_max,
        self.torque_noise_max
      )

    _, dx = self.step(torque, timestep)

    self.past_state_pushback(traj, dx)

    terminal = self._terminal(dx)
    reward = 1 - (1 / self._state_error(traj, dx)) if not terminal else 2
    return (self._get_ob(), reward, terminal, {})

  def _get_ob(self):
    # s = self.state
    # return np.array([cos(s[0]), sin(s[0]), cos(s[1]), sin(s[1]), s[2], s[3]])
    return self.state[:2]

  def _terminal(self, state):

    errors = np.array(self.past_error.values())
    if np.mean(errors) < 1:
      return True

    return False

  def render(self, mode='human'):
    from gym.envs.classic_control import rendering

    s = self.state

    if self.viewer is None:
      self.viewer = rendering.Viewer(500, 500)
      bound = self.LINK_LENGTH_1 + self.LINK_LENGTH_2 + 0.2  # 2.2 for default
      self.viewer.set_bounds(-bound, bound, -bound, bound)

    if s is None:
      return None

    p1 = [self.LINK_LENGTH_1 * cos(s[0]), -self.LINK_LENGTH_1 * sin(s[0])]

    p2 = [
      p1[0] + self.LINK_LENGTH_2 * cos(s[0] + s[1]),
      p1[1] - self.LINK_LENGTH_2 * sin(s[0] + s[1])
    ]

    xys = np.array([[0, 0], p1, p2])[:, ::-1]
    thetas = [s[0], s[0] + s[1]]
    link_lengths = [self.LINK_LENGTH_1, self.LINK_LENGTH_2]

    self.viewer.draw_line((-2.2, 1), (2.2, 1))
    for ((x, y), th, llen) in zip(xys, thetas, link_lengths):
      l, r, t, b = 0, llen, .1, -.1
      jtransform = rendering.Transform(rotation=th, translation=(x, y))
      link = self.viewer.draw_polygon([(l, b), (l, t), (r, t), (r, b)])
      link.add_attr(jtransform)
      link.set_color(0, .8, .8)
      circ = self.viewer.draw_circle(.1)
      circ.set_color(.8, .8, 0)
      circ.add_attr(jtransform)

    return self.viewer.render(return_rgb_array=mode == 'rgb_array')

  def close(self):
    if self.viewer:
      self.viewer.close()
      self.viewer = None

  # Redifining controller
  def generate_system(self):

    i = 0.0
    count = 0
    scale = 1e3
    Kp = 1 * np.eye(2) * scale
    Kv = 1e-2 * Kp

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
      prev_time, dx = self.env_step(u)

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

      print(err[0])

      # if count % 100 == 0:
      #   plt.plot(J1, 'r')
      #   plt.plot(J2, 'b')
      #   plt.plot(T1, 'r--')
      #   plt.plot(T2, 'b--')
      #   plt.show()
      #   plt.pause(1e-4)
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


if __name__ == '__main__':
  # Tests
  rl_controller = RLController(input_shape=(2, 1))
  rl_controller.generate_system()