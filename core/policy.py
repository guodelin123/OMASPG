import numpy as np
import tensorflow as tf
from common.utils import fc
from common.distributions import DiagGaussianPdType


class GaussianPolicy:

    def __init__(self, ob_space, ac_space, n_node=256, name='pi', squash=True, fixed_noise=None):
        self.ob_dim = ob_space.shape[0]  # ob dim
        self.ac_dim = ac_space.shape[0]  # ac dim
        self.squash = squash
        self.name = name
        self.n_node = n_node
        self.fixed_noise = fixed_noise

        self.obs_ph = tf.placeholder(dtype=tf.float32, shape=(None, self.ob_dim))
        self.actions, _,  _, self.dist = self.forward(self.obs_ph)

    def forward(self, observations):
        with tf.variable_scope(self.name, reuse=tf.AUTO_REUSE):
            dist = Normal(self.ac_dim, observations, self.n_node, self.fixed_noise)  # distribution
        raw_actions = dist.act
        actions = tf.tanh(raw_actions) if self.squash else raw_actions
        log_ps = dist.log_p
        if self.squash:
            log_ps -= self.squash_correction(raw_actions)
        return actions, raw_actions, log_ps, dist

    def get_actions(self, observations, is_deterministic=False):
        feed_dict = {self.obs_ph: observations}
        if is_deterministic:
            mu = tf.get_default_session().run(self.dist.mean, feed_dict)
            if self.squash:
                mu = np.tanh(mu)
            return mu
        else:
            return tf.get_default_session().run(self.actions, feed_dict)

    def get_action(self, observation, is_deterministic=False):
        return self.get_actions(observation[None], is_deterministic)[0]

    def squash_correction(self, actions):
        if not self.squash:
            return 0.
        return tf.reduce_sum(tf.log(1 - tf.tanh(actions) ** 2 + 1e-6), axis=1)

    def get_params(self):
        scope = tf.get_variable_scope().name
        scope += '/' + self.name + '/' if len(scope) else self.name + '/'
        return tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=scope)


class Normal(object):
    def __init__(self, act_dim, input_ph, n_node=256, fixed_noise=None):
        self.input_ph = input_ph
        self.act_dim = act_dim
        self.n_node = n_node
        self.fixed_noise = fixed_noise
        self._create_graph()

    def _create_graph(self):
        with tf.variable_scope(name_or_scope='mlp', reuse=tf.AUTO_REUSE):
            activ = tf.nn.relu
            h1 = activ(fc(self.input_ph, 'pi_fc1', nh=self.n_node, init_scale=0.01))
            h2 = activ(fc(h1, 'pi_fc2', nh=self.n_node, init_scale=0.01))
            if self.fixed_noise is not None:
                mu_t = fc(h2, 'pi', self.act_dim, init_scale=0.01)
                logsig_t = tf.constant(dtype=tf.float32, value=np.log(self.fixed_noise), shape=[1, self.act_dim])
                mu_and_logsig_t = tf.concat([mu_t, 0.0 * mu_t + logsig_t], axis=-1)
            else:
                mu_and_logsig_t = fc(h2, 'pi', 2 * self.act_dim, init_scale=0.01)

        pdtype = DiagGaussianPdType(self.act_dim)
        self.pd = pdtype.pdfromflat(mu_and_logsig_t)
        self.mean = self.pd.mean
        self.logstd = self.pd.logstd
        self.act = self.pd.sample()
        self.log_p = self.pd.logp(self.act)

