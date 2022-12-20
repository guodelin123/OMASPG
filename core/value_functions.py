import tensorflow as tf
import numpy as np
from common.utils import fc


class Q_Function:
    def __init__(self, env, n_node=256, name='q_func'):
        self.name = name
        self.n_node = n_node
        self.state_dim = env.state_space.shape[0]
        self.nagent = env.nagent

        self.state_ph = tf.placeholder(tf.float32, shape=[None, self.state_dim])
        self.act_ph_n = [tf.placeholder(tf.float32, shape=[None, env.action_space[i].shape[0]])
                         for i in range(self.nagent)]
        self.qf = self.forward(self.state_ph, self.act_ph_n)

    def forward(self, states, acts):
        q_input = tf.concat([states] + acts, axis=-1)
        with tf.variable_scope(self.name, reuse=tf.AUTO_REUSE):
            activ = tf.nn.relu
            h1 = activ(fc(q_input, 'qf_fc1', nh=self.n_node, init_scale=np.sqrt(2)))
            h2 = activ(fc(h1, 'qf_fc2', nh=self.n_node, init_scale=np.sqrt(2)))
            qf = fc(h2, 'qf', 1)[:, 0]
        return qf

    def get_value(self, states, acts):
        if states.shape.__len__() == 1:
            states = states[None]
            acts = [act[None] for act in acts]
        feed_dict = {self.act_ph_n[i]: acts[i] for i in range(self.nagent)}
        feed_dict[self.state_ph] = states
        return tf.get_default_session().run(self.qf, feed_dict)

    def get_params(self):
        scope = tf.get_variable_scope().name
        scope += '/' + self.name + '/' if len(scope) else self.name + '/'
        return tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=scope)
