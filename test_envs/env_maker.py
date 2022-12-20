import gym
import numpy as np
from gym.spaces import Box
from multiagent_mujoco.mujoco_multi import MujocoMulti

class MaMujocoEnv(gym.Env):
    def __init__(self,
                 scenario="Ant-v2",
                 agent_conf='2x4',
                 agent_obsk=None,
                 episode_limit=1000,
                 ):
        env_args = {'scenario': scenario, 'agent_conf': agent_conf,
                    'agent_obsk': agent_obsk, 'episode_limit': episode_limit}
        self.env = MujocoMulti(env_args=env_args)

        self.scenario = scenario
        self.agent_obsk = agent_obsk
        self.agent_conf = agent_conf
        self.episode_limit = episode_limit

        env_info = self.env.get_env_info()
        self.nagent = env_info['n_agents']

        ob_dim = env_info['obs_shape']
        state_dim = env_info['state_shape']
        self.observation_space = [Box(low=-np.ones(ob_dim), high=np.ones(ob_dim), dtype=np.float64) for _ in
                                  range(self.nagent)]
        self.state_space = Box(low=-np.ones(state_dim), high=np.ones(state_dim), dtype=np.float64)
        self.action_space = self.env.action_space

    def reset(self):
        self.env.reset()
        ob_n = self.env.get_obs()
        state = self.env.get_state()
        return np.asarray(ob_n), state

    def step(self, actions):
        reward, terminated, env_info = self.env.step(actions)
        ob_n = self.env.get_obs()
        state = self.env.get_state()
        return ob_n, state, reward, terminated, env_info

    def render(self, mode="human"):
        self.env.render()


if __name__ == "__main__":
    env = MaMujocoEnv(scenario="Humanoid-v2",
                      agent_conf='9|8',
                      agent_obsk=None,
                      episode_limit=1000)
    ob_n, state = env.reset()
