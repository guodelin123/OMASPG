import numpy as np


class ReplayBuffer:
    def __init__(self, env, max_buffer_size):

        self.max_buffer_size = max_buffer_size = int(max_buffer_size)
        self.nagent = env.nagent
        agent_obsk = getattr(env, 'agent_obsk', 'null')
        if agent_obsk is None:
            self.full_ob = True
        else:
            self.full_ob = False
        if not self.full_ob:
            self.obs_n = [np.zeros((max_buffer_size, env.observation_space[i].shape[0]), dtype=np.float32)
                          for i in range(self.nagent)]
            self.next_obs_n = [np.zeros((max_buffer_size, env.observation_space[i].shape[0]), dtype=np.float32)
                          for i in range(self.nagent)]
        self.acts_n = [np.zeros((max_buffer_size, env.action_space[i].shape[0])) for i in range(self.nagent)]
        self.states = np.zeros((max_buffer_size, env.state_space.shape[0]), dtype=np.float32)
        self.next_states = np.zeros((max_buffer_size, env.state_space.shape[0]), dtype=np.float32)
        self.rewards = np.zeros(max_buffer_size, dtype=np.float32)
        self.masks = np.zeros(max_buffer_size)
        self.top = 0
        self._size = 0

    def add_sample(self, ob_n, next_ob_n, act_n, state, next_state, reward, mask):
        for i in range(self.nagent):
            if not self.full_ob:
                self.obs_n[i][self.top] = ob_n[i]
                self.next_obs_n[i][self.top] = next_ob_n[i]
            self.acts_n[i][self.top] = act_n[i]
        self.rewards[self.top] = reward
        self.states[self.top] = state
        self.next_states[self.top] = next_state
        self.masks[self.top] = mask
        self.top = (self.top + 1) % self.max_buffer_size
        if self._size < self.max_buffer_size:
            self._size += 1

    def random_batch(self, batch_size):
        indices = np.random.randint(0, self.size, batch_size)
        states = self.states[indices]
        next_states = self.next_states[indices]
        if not self.full_ob:
            obs_n = [self.obs_n[i][indices] for i in range(self.nagent)]
            next_obs_n = [self.next_obs_n[i][indices] for i in range(self.nagent)]
        else:
            obs_n = [states for _ in range(self.nagent)]
            next_obs_n = [next_states for _ in range(self.nagent)]
        return dict(obs_n=obs_n,
                    next_obs_n = next_obs_n,
                    acts_n=[self.acts_n[i][indices] for i in range(self.nagent)],
                    rewards=self.rewards[indices],
                    states=states,
                    next_states=next_states,
                    masks=self.masks[indices])

    @property
    def size(self):
        return self._size
