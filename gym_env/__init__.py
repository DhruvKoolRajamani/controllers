from gym.envs.registration import register
from .envs.orthosis_env import OrthosisEnv

register(
  id='orthosis-v0',
  entry_point='gym_env.envs:OrthosisEnv',
)
