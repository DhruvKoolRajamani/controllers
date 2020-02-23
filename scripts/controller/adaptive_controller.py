import scipy.integrate as integrate
import numpy as np
from numpy import sin, cos, pi
from controller.controller_extensions import *
import matplotlib.pyplot as plt
from controller.orthosis import Orthosis
from scipy.linalg.blas import sgemm


class AdaptiveController(Orthosis):
  """ Adaptive Controller inherited from Orthosis """

  def __init__(
    self,
    x0=[0.,
        0.,
        0.,
        0.],
    input_shape=(1,
                 1),
    timestep=1e-7,
    tf=2,
    alpha0=[0.]
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
      *alpha0*
          Initial state matrix for regressor coeffs
    """

    super(AdaptiveController,
          self).__init__(
            x0=x0,
            input_shape=input_shape,
            timestep=timestep,
            model='adaptive',
            tf=5
          )

    self.alpha = np.array(alpha0)

    return

  def rbf(self, X, var=1, gamma=1):
    """ """
    for i in X:
      X[i] = var * np.exp(-gamma * X[i]**2)
    return X

  # Redifining controller
  def generate_system(self):

    Y = None
    i = 0.0
    count = 0
    scale = 1
    Kp = 100 * np.eye(2) * scale
    Kv = 1e-2 * Kp
    decay = 0

    traj_coeffs = np.array(
      [
        (self.JOINT_LIMITS_1[1] - self.JOINT_LIMITS_1[0]) / 10,
        (self.JOINT_LIMITS_2[1] - self.JOINT_LIMITS_2[0]) / 10
      ]
    ).reshape(2,
              )

    lam = 2 * Kp
    L = 50 * np.eye(12) * scale
    self.alpha = np.array(
      [
        0.1,
        0.01,
        0.001,
        0.0001,
        0.00001,
        0.000001,
        0.1,
        0.01,
        0.001,
        0.0001,
        0.00001,
        0.000001
      ]
    ).reshape(12,
              1)

    s = None

    U1 = []
    U2 = []
    J1 = []
    J2 = []
    T1 = []
    T2 = []
    E1 = []
    E2 = []
    t = []

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
      # print(err)

      # Defining sliding surface
      sld1 = err[:2]
      sld1 = sld1.reshape(2, 1)
      sld2 = err[2:]
      sld2 = sld2.reshape(2, 1)

      sld = sld1 + lam.dot(sld2)
      print(sld)

      x_unk_1 = np.array(
        [
          1,
          self.state[0],
          self.state[0]**2,
          self.state[0]**3,
          self.state[0]**4,
          self.state[0]**5
        ]
      )
      x_unk_2 = np.array(
        [
          1,
          self.state[1],
          self.state[1]**2,
          self.state[1]**3,
          self.state[1]**4,
          self.state[1]**5
        ]
      )

      # Consider using RBF but not necessary
      Y = np.zeros((2, 12))
      Y[0][:6] = x_unk_1
      Y[1][6:] = x_unk_2

      alpha_grad = np.linalg.inv(-L).dot(Y.T.dot(Kp.dot(sld)))
      print(alpha_grad)

      # self.alpha = self.alpha.reshape(12, 1)
      if i == 0:
        decay = 0
        self.alpha = self.alpha
      else:
        decay = i - decay
        self.alpha += alpha_grad * decay

      u = Y.dot(self.alpha.reshape(12, 1)) - Kp.dot(sld)

      prev_time, dx = self.step(u, self.timestep)

      U1.append(u[0][0])
      U2.append(u[1][0])
      J1.append(dx[0])
      J2.append(dx[1])
      t.append(prev_time)
      T1.append(traj[0])
      T2.append(traj[1])
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
  adaptive_controller = AdaptiveController(input_shape=(2, 1))
  adaptive_controller.generate_system()