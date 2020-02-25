import scipy.integrate as integrate
import numpy as np
from numpy import sin, cos, pi
from controller.orthosis import Orthosis
import matplotlib.pyplot as plt
import controller.controller_extensions

from gym import core, spaces
from gym.utils import seeding


class OrthosisEnv(Orthosis, core.Env):
  """ Adaptive Controller inherited from Orthosis """

  metadata = {
    'render.modes': ['human',
                     'rgb_array'],
    'video.frames_per_second': 15
  }

  # MODIFY THIS TO CHANGE THE RESPONSE. A SMALLER VALUE WILL MAKE IT SMOOTHER
  # AND SHOW MORE STEPS. LARGER VALUE WILL MAKE IT MORE STIFF.
  dt = 1e-4
  AVAIL_TORQUE = [-0.00001, 0., +0.00001]

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

    super(OrthosisEnv,
          self).__init__(
            x0=x0,
            input_shape=input_shape,
            model='rl',
            timestep=self.dt
          )
    self.viewer = None

    high = np.zeros(self.JOINT_LIMITS_1.shape)
    high[0] = self.JOINT_LIMITS_1[1]
    high[1] = self.JOINT_LIMITS_2[1]
    high[2] = self.JOINT_LIMITS_1[3]
    high[3] = self.JOINT_LIMITS_2[3]

    low = np.zeros(self.JOINT_LIMITS_1.shape)
    low[0] = self.JOINT_LIMITS_1[0]
    low[1] = self.JOINT_LIMITS_2[0]
    low[2] = self.JOINT_LIMITS_1[2]
    low[3] = self.JOINT_LIMITS_2[2]

    self.beta = None
    self.prev_time = None

    self.env_reset = True

    self.traj = np.zeros((4,))

    # Set higher number of states to be checked for a reward
    self.past_max = 10000

    # store at least past_max past values of trajectories and states
    self.past_trajectories = np.zeros((self.past_max, 4))
    self.past_states = np.zeros((self.past_max, 4))
    self.past_error = None
    if self.input_shape == (1, 1):
      self.past_error = np.zeros((self.past_max, 1))
    else:
      self.past_error = np.zeros((self.past_max, 2))
    self.past_count = 0

    self.observation_space = spaces.Box(low=low, high=high, dtype=np.float32)

    # Define 1 or two possible action spaces if input is 1,1 or 2,1
    self.action_space = None

    if self.input_shape == (1, 1):
      self.action_space = np.array([spaces.Discrete(3)])
    else:
      self.action_space = np.array([spaces.Discrete(3), spaces.Discrete(3)])

    self.state = x0
    self.seed()

    return

  def seed(self, seed=None):
    self.np_random, seed = seeding.np_random(seed)
    return [seed]

  def reset(self):
    self.state = self.np_random.uniform(
      low=np.array([0.1,
                    0.1,
                    0.,
                    0.]),
      high=np.array(
        [0.7 * self.JOINT_LIMITS_1[1],
         0.7 * self.JOINT_LIMITS_2[1],
         0.,
         0.]
      ),
      size=(4,
            )
    )
    th = None
    coeff = None
    if self.exclusive_traj == 1:
      th = self.state[0]
      coeff = self.traj_coeffs[0]
    elif self.exclusive_traj == 2:
      th = self.state[1]
      coeff = self.traj_coeffs[1]

    self.traj = np.zeros((4,))
    self.alpha = pi / 2
    self.beta = th / coeff
    self.prev_time = 0.
    traj = self._desired_trajectory(
      t=self.prev_time,
      coeffs=self.traj_coeffs,
      traj_type='sin',
      frequency=10 * self.scale,
      stiff_traj=False,
      alpha=self.alpha,
      beta=self.beta
    )
    self.traj = traj
    self.env_reset = True
    print(self.traj)
    self.past_trajectories = np.zeros((self.past_max, 4))
    self.past_states = np.zeros((self.past_max, 4))
    if self.input_shape == (1, 1):
      self.past_error = np.zeros((self.past_max, 1))
    else:
      self.past_error = np.zeros((self.past_max, 2))
    self.past_count = 0
    self.count = 0

    return self._get_ob()

  def past_state_pushback(self, traj, state, err):
    """ Don't need to touch this """

    if self.past_count < self.past_max:
      self.past_states[self.past_count] = state.reshape(1, 4)
      self.past_trajectories[self.past_count] = traj.reshape(1, 4)
      if self.input_shape == (1, 1):
        self.past_error[self.past_count] = err
      else:
        self.past_error[self.past_count] = err.reshape(1, 2)
      self.past_count += 1
    else:
      self.past_states = self.past_states[1:]
      self.past_states = np.append(
        self.past_states,
        state.reshape(1,
                      4),
        axis=0
      )
      self.past_trajectories = self.past_trajectories[1:, :]
      self.past_trajectories = np.append(
        self.past_trajectories,
        traj.reshape(1,
                     4),
        axis=0
      )
      if self.input_shape == (1, 1):
        self.past_error = self.past_error[1:]
        self.past_error = np.append(self.past_error, err)
      else:
        self.past_error = self.past_error[1:, :]
        self.past_error = np.append(self.past_error, err.reshape(1, 2), axis=0)

    return

  def _state_error(self, traj, state):
    """ Calculates the error for each state. Definitely modifiable based on 
    what you want to use as error """

    max_joint_error = np.zeros((2,))
    err = 0.
    max_joint_error[0] = self.JOINT_LIMITS_1[1] - self.JOINT_LIMITS_1[0]
    max_joint_error[1] = self.JOINT_LIMITS_2[1] - self.JOINT_LIMITS_2[0]

    error = traj - state
    # print(error)

    err = None
    i = -1
    if self.input_shape == (1, 1):
      if self.exclusive_traj == 1:
        err = error[0]
        i = 0
        #/ max_joint_error[0] + 2 * error[1] / max_joint_error[1]
      elif self.exclusive_traj == 2:
        err = 2 * error[0] / max_joint_error[0] + 0.1 * error[
          1] / max_joint_error[1]
        i = 1
      else:
        err = error[0] / max_joint_error[0] + error[1] / max_joint_error[1]
    else:
      err = np.array(
        [error[0] / max_joint_error[0],
         error[1] / max_joint_error[1]]
      )
    # print(err)
    rew = None
    if self.prev_time < self.dt * 100:
      rew = -np.absolute(max_joint_error[i] / 2)
    else:
      rew = -np.absolute(err)

    return rew, err

  def env_step(self, a, timestep=0.01):

    if not self.env_reset:
      traj = self._desired_trajectory(
        t=self.prev_time,
        coeffs=self.traj_coeffs,
        traj_type='sin',
        frequency=10 * self.scale,
        stiff_traj=False,
        alpha=self.alpha,
        beta=self.beta
      )
      self.traj = traj

    torque = None

    if self.input_shape == (1, 1):
      torque = self.AVAIL_TORQUE[a]
    else:
      torque = np.array([self.AVAIL_TORQUE[a[0]], self.AVAIL_TORQUE[a[1]]])

    if self.torque_noise_max > 0:
      torque += self.np_random.uniform(
        -self.torque_noise_max,
        self.torque_noise_max
      )

    cnt, dx = self.step(torque, self.dt)
    self.prev_time += cnt

    reward, err = self._state_error(self.traj, dx)

    self.past_state_pushback(self.traj, dx, reward)

    terminal = self._terminal(dx)
    if self.input_shape == (1, 1):
      reward = reward if not terminal else 100
    else:
      reward = reward if not terminal else [100, 100]

    self.env_reset = False
    # print(reward)
    # print(self.prev_time)
    # IF MULTI INPUT MAYBE RESET WHEN BOTH REWARDS RETURN 100 AND NOT JUST ONE
    return (self._get_ob(), reward, terminal, {})

  def _get_ob(self):
    # s = self.state
    # return np.array([cos(s[0]), sin(s[0]), cos(s[1]), sin(s[1]), s[2], s[3]])
    return self.state

  def _terminal(self, state):
    thresh = None
    if self.past_count == self.past_max:
      mean = None

      if self.input_shape == (1, 1):
        thresh = 0.002
        mean = np.absolute(np.mean(self.past_error))
        mean = mean - thresh
        if mean > 0:
          return False
        else:
          return True
        # if np.all(self.past_error):
        #   return True
      else:
        thresh = np.array([0.1, 0.1])
        mean = np.absolute(np.mean(self.past_error, axis=0))
        mean = mean - thresh
        # print(mean)
        # If any item is larger than threshold, then all errors have not converged
        if np.any(mean):
          return False
        else:
          return True

    return False

  def get_current_data(self):
    """ Function to return the current state and trajectory data """

    d = {'state': self.state[:2], 'traj': self.traj[:2]}

    return d

  # NOT USING THIS
  def render(self, mode='human'):
    from gym.envs.classic_control import rendering

    s = self.state

    if self.viewer is None:
      self.viewer = rendering.Viewer(500, 500)
      bound = self.LINK_LENGTH_1 + self.LINK_LENGTH_2 + 0.2  # 2.2 for default
      self.viewer.set_bounds(-bound, bound, -bound, bound)

    if s is None:
      return None

    p1 = [-self.LINK_LENGTH_1 * cos(s[0]), self.LINK_LENGTH_1 * sin(s[0])]

    p2 = [
      p1[0] - self.LINK_LENGTH_2 * cos(s[0] + s[1]),
      p1[1] + self.LINK_LENGTH_2 * sin(s[0] + s[1])
    ]

    xys = np.array([[0, 0], p1, p2])[:, ::-1]
    thetas = [s[0] - pi/2, s[0] + s[1] - pi/2]
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


if __name__ == '__main__':
  # Tests
  controller = OrthosisEnv(input_shape=(1, 1))
  controller.env_step(0)
