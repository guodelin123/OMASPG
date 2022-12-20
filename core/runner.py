import numpy as np
from common import logger
import time

from core.replay_buffer import ReplayBuffer


class Runner:

    def __init__(self, env, get_action, args):
        self.env = env
        self.nagent = env.nagent
        self.get_action = get_action
        self.replay_buffer = ReplayBuffer(env, args.max_buffer_size)
        self.r_scale = getattr(args,'r_scale', 1.)
        self.min_buffer_size = args.min_buffer_size
        self.batch_size = args.batch_size
        self.episode_limit = env.episode_limit
        self.path_length = 0
        self.path_return = 0
        self.last_path_return = 0
        self.max_path_return = -np.inf
        self.n_episodes = 0
        self.cur_ob_n = None
        self.cur_state = None
        self.total_samples = 0

    def sample(self):
        if self.cur_ob_n is None:
            self.cur_ob_n, self.cur_state = self.env.reset()
        act_n = self.get_action(self.cur_ob_n)
        next_ob_n, next_state, reward, terminal, info = self.env.step(act_n)

        self.path_length += 1
        self.path_return += reward
        self.total_samples += 1

        # if True, terminal is caused by episode limitation
        # so the next state should not be masked
        episode_limit = info.get("episode_limit", False)

        if episode_limit:
            mask = False
        else:
            mask = terminal

        self.replay_buffer.add_sample(ob_n=self.cur_ob_n,
                                      next_ob_n=next_ob_n,
                                      act_n=act_n,
                                      state=self.cur_state,
                                      next_state=next_state,
                                      reward=reward * self.r_scale,
                                      mask=mask)

        # In MaMuJoCo, terminal is True when the task is really done or the
        # timestep limitation is reached.
        if terminal:
            self.cur_ob_n, self.cur_state = self.env.reset()
            self.path_length = 0
            self.max_path_return = max(self.max_path_return,
                                       self.path_return)
            self.last_path_return = self.path_return
            self.path_return = 0
            self.n_episodes += 1

        else:
            self.cur_ob_n = next_ob_n
            self.cur_state = next_state

    def batch_ready(self):
        enough_samples = (self.replay_buffer.size >= self.min_buffer_size)
        return enough_samples

    def random_batch(self):
        return self.replay_buffer.random_batch(self.batch_size)

    def log_diagnostics(self):
        logger.logkv('buffer-size', self.replay_buffer.size)
        logger.logkv('max-path-return', self.max_path_return)
        logger.logkv('last-path-return', self.last_path_return)
        logger.logkv('episodes', self.n_episodes)
        logger.logkv('total-samples', self.total_samples)


def rollout(env, get_action, det=True, sleep_time=0.01, render=False):
    ob_n, _ = env.reset()
    path_length = env.episode_limit
    rewards = np.zeros((path_length,))
    t = 0
    for t in range(path_length):
        act_n = get_action(ob_n, det)
        next_ob_n, next_state, reward, terminal, info = env.step(act_n)
        rewards[t] = reward
        ob_n = next_ob_n
        if render:
            env.render()
            time.sleep(sleep_time)
        if terminal:
            break
    return rewards[:t + 1]
