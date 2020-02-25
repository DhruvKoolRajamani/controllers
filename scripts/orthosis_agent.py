import argparse
import sys
import numpy as np

import gym
import gym_env
from gym import wrappers, logger

import matplotlib.pyplot as plt


class OrthosisAgent(object):
  """The world's simplest agent!"""

  def __init__(self, action_space):
    self.action_space = action_space

  # Havent figured out how to return an array of samples
  def act(self, observation, reward, done):
    # Add code here
    # TRY CROSS ENTROPY?
    act = self.action_space[0].sample(
    ) if self.action_space.size == 1 else np.array(
      [self.action_space[0].sample(),
       self.action_space[1].sample()]
    )
    return act


if __name__ == '__main__':
  parser = argparse.ArgumentParser(description=None)
  parser.add_argument(
    'env_id',
    nargs='?',
    default='orthosis-v0',
    help='Select the environment to run'
  )
  args = parser.parse_args()

  # You can set the level to logger.DEBUG or logger.WARN if you
  # want to change the amount of output.
  logger.set_level(logger.INFO)

  env = gym.make(args.env_id)

  # You provide the directory to write to (can be an existing
  # directory, including one with existing data -- all monitor files
  # will be namespaced). You can also dump to a tempdir if you'd
  # like: tempfile.mkdtemp().
  outdir = '/tmp/orthosis-agent-results'
  env = wrappers.Monitor(
    env,
    directory=outdir,
    force=True,
    video_callable=False
  )
  env.seed(0)
  agent = OrthosisAgent(env.action_space)

  episode_count = 1000
  reward = 0
  done = False

  fig = plt.figure()
  env.set_exclusive_traj(1)

  for i in range(episode_count):
    ob = env.reset()

    # ds for plotting
    J1 = []
    J2 = []
    T1 = []
    T2 = []
    R = []
    Action = []
    plot_counter = 0
    plt.ion()
    fig = plt.gcf()
    ax1 = fig.add_subplot(211)
    ax1.set_title('Joint Positions and Trajectories')

    ax2 = fig.add_subplot(212)
    ax2.set_title('Rewards')

    ax1.plot(J1, 'r', label='MCP Joint')
    ax1.plot(J2, 'b', label='PIP Joint')
    ax1.plot(T1, 'r--', label='MCP Trajectory')
    ax1.plot(T2, 'b--', label='PIP Trajectory')
    ax1.legend()
    ax2.plot(R, 'g', label='Reward')
    ax2.legend()
    plt.show()
    plt.pause(1e-4)

    while True:
      action = agent.act(ob, reward, done)
      ob, reward, done, _ = env.env_step(action)
      d = env.get_current_data()
      T1.append(d['traj'][0])
      T2.append(d['traj'][1])
      J1.append(d['state'][0])
      J2.append(d['state'][1])
      R.append(reward)
      Action.append(action)
      plot_counter += 1

      # Call plotting every 100 counts,
      # can increase this if performace is slow
      if plot_counter % 1000 == 0:

        ax1.plot(J1, 'r')
        ax1.plot(J2, 'b')
        ax1.plot(T1, 'r--')
        ax1.plot(T2, 'b--')
        ax1.legend()
        ax2.plot(R, 'g')
        ax2.legend()
        plt.show()
        plt.pause(1e-4)

      # print(env.prev_time)
      if env.prev_time >= 10:
        plt.clf()
        break

      if done:
        plt.clf()
        break
      # Note there's no env.render() here. But the environment still can open window and
      # render if asked by env.monitor: it calls env.render('rgb_array') to record video.
      # Video is not recorded every episode, see capped_cubic_video_schedule for details.

  # Close the env and write monitor result info to disk
  env.close()
