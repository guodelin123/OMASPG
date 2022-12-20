import numpy as np
import tensorflow as tf
from common import logger
from core.runner import rollout
import joblib
import os.path as osp
import os
import common.utils as U
from core.policy import GaussianPolicy
from core.value_functions import Q_Function
from core.runner import Runner
import time

"""The algorithm"""
class Model:
    def __init__(self, env, eval_env, args):
        self.env = env
        self.eval_env = eval_env
        self.nagent = self.env.nagent

        self.n_epochs = args.n_epochs
        self.update_interval = args.update_interval
        self.n_train_repeat = args.n_train_repeat
        self.epoch_length = args.epoch_length
        self.total_time_steps = self.n_epochs * self.epoch_length
        self.eval_n_episodes = args.eval_n_episodes
        self.gamma = args.gamma
        self.target_update_interval = args.target_update_interval
        self.save_interval = args.save_interval
        self.squash = args.squash

        self.max_grad_norm = args.max_grad_norm
        self.huber_delta = args.huber_delta
        self.two_qf = getattr(args, 'two_qf', True)

        self.pi_lr = args.pi_lr
        self.critic_lr = args.critic_lr
        self.alpha_lr = args.alpha_lr
        self.beta_lr = args.beta_lr

        self.actor_tau = args.actor_tau
        self.critic_tau = args.critic_tau

        #  actor, critic optimizers
        self.pi_LR = tf.placeholder(tf.float32, shape=[], name='pi_lr')
        self.critic_LR = tf.placeholder(tf.float32, shape=[], name='critic_lr')
        self.pi_optimizer = tf.train.AdamOptimizer(self.pi_LR)
        self.critic_optimizer = tf.train.AdamOptimizer(self.critic_LR)
        self.pi_lr_decay = getattr(args, 'pi_lr_decay', False)
        self.critic_lr_decay = getattr(args, 'critic_lr_decay', False)

        # alpha, entropy coefficient
        self.alpha_optimizer = tf.train.AdamOptimizer(self.alpha_lr)
        self.auto_alpha = args.auto_alpha
        self.target_entropy = -np.sum([env.action_space[i].shape[0] for i in range(self.nagent)])
        self.log_alpha = tf.Variable(initial_value=0., name='log_alpha',
                                     constraint=lambda z: tf.clip_by_value(z, -12, 5))
        if self.auto_alpha:
            self.alpha = tf.exp(self.log_alpha, name='alpha')
        else:
            self.alpha = tf.constant(value=getattr(args, 'alpha', 0.1))

        # trpo,  KL penalty
        self.trpo = args.trpo
        beta_optimizer = getattr(args, 'beta_optimizer', None)
        if beta_optimizer == 'adam':
            self.beta_optimizer = tf.train.AdamOptimizer(self.beta_lr)
        elif beta_optimizer == 'sgd':
            self.beta_optimizer = tf.train.GradientDescentOptimizer(self.beta_lr)
        else:
            raise NotImplementedError
        self.delta_n = [tf.constant(args.delta_base) * env.action_space[i].shape[0] for i in range(self.nagent)]
        self.log_beta_n = [tf.Variable(0., constraint=lambda z: tf.clip_by_value(z, -20, np.log(100)))
                           for _ in range(self.nagent)]
        self.beta_n = [tf.exp(log_beta) for log_beta in self.log_beta_n]

        # centralized critics
        self.qf1 = Q_Function(env, args.n_node, name='qf1')
        self.qf2 = Q_Function(env, args.n_node, name='qf2')

        # decentralized polices
        self.ob_space_n = env.observation_space
        self.ac_space_n = env.action_space

        self.pi_n = [GaussianPolicy(self.ob_space_n[i], self.ac_space_n[i], args.n_node,
                                    name='pi_%d' % i, squash=self.squash)
                     for i in range(self.nagent)]

        self.oldpi_n = [GaussianPolicy(self.ob_space_n[i], self.ac_space_n[i], args.n_node,
                                       name='old_pi_%d' % i, squash=self.squash)
                        for i in range(self.nagent)]

        self.runner = Runner(env, self.get_action, args)

        ###
        self.critic_train_ops = []
        self.pi_train_op_n = []
        self.diagnosis_op = []
        self.loss_names = ['td_loss1', 'td_loss2', 'approx_ent', 'alpha']
        if not self.two_qf:
            self.loss_names.pop(1)

        self._init_placeholders()
        self._init_critic_update()
        self._init_actor_update()
        self._init_target_ops()

        self.sess = tf.get_default_session()
        self.sess.run(tf.global_variables_initializer())
        self.update_target(1., 1.)  # assign new to old

    def _init_placeholders(self):

        self.obs_ph_n, self.next_obs_ph_n, self.acts_ph_n = [], [], []
        for i in range(self.nagent):
            self.obs_ph_n.append(tf.placeholder(tf.float32, shape=(None, self.ob_space_n[i].shape[0]), name='ob%d' % i))
            self.next_obs_ph_n.append(
                tf.placeholder(tf.float32, shape=(None, self.ob_space_n[i].shape[0]), name='next_ob%d' % i))
            self.acts_ph_n.append(
                tf.placeholder(tf.float32, shape=(None, self.ac_space_n[i].shape[0]), name='act%d' % i))

        self.state_dim = self.env.state_space.shape[0]
        self.states_ph = tf.placeholder(tf.float32, shape=(None, self.state_dim), name='states')
        self.next_states_ph = tf.placeholder(tf.float32, shape=(None, self.state_dim), name='next_ob')
        self.rewards_ph = tf.placeholder(tf.float32, shape=(None,), name='rewards')
        self.masks_ph = tf.placeholder(tf.float32, shape=(None,), name='masks')
        self.scale_reward_ph = tf.placeholder(tf.float32, shape=(), name='scale_reward')

    def _init_critic_update(self):
        self.qf1_t = self.qf1.forward(self.states_ph, self.acts_ph_n)
        self.qf2_t = self.qf2.forward(self.states_ph, self.acts_ph_n)

        next_acts_n, next_logps_n = [], []
        for i in range(self.nagent):
            next_acts, _, next_logps, _ = self.oldpi_n[i].forward(self.next_obs_ph_n[i])
            next_acts_n.append(next_acts)
            next_logps_n.append(next_logps)

        with tf.variable_scope('target'):
            qf1_next_target_t = self.qf1.forward(self.next_states_ph, next_acts_n)
            qf2_next_target_t = self.qf2.forward(self.next_states_ph, next_acts_n)
            self.qf1_target_params = self.qf1.get_params()
            self.qf2_target_params = self.qf2.get_params()

        old_next_logpas_joint = tf.reduce_sum(
            tf.concat([next_logps_n[i][..., None] for i in range(self.nagent)], axis=-1),
            axis=-1)

        if self.two_qf:
            qf_next_target = tf.minimum(qf1_next_target_t, qf2_next_target_t) - self.alpha * old_next_logpas_joint
        else:
            qf_next_target = qf1_next_target_t - self.alpha * old_next_logpas_joint

        ys = tf.stop_gradient(self.rewards_ph + (1 - self.masks_ph) * self.gamma * qf_next_target)

        if self.huber_delta is not None:
            self.td_loss1_t = tf.losses.huber_loss(ys, self.qf1_t, delta=self.huber_delta)
            self.td_loss2_t = tf.losses.huber_loss(ys, self.qf2_t, delta=self.huber_delta)
        else:
            self.td_loss1_t = 0.5 * tf.losses.mean_squared_error(ys, self.qf1_t)
            self.td_loss2_t = 0.5 * tf.losses.mean_squared_error(ys, self.qf2_t)

        def minimize(loss, vars_list, max_grad_norm=None):
            grads = tf.gradients(loss, vars_list)
            if max_grad_norm is not None:
                grads, _ = tf.clip_by_global_norm(grads, max_grad_norm)  # clip gradients
            grads = list(zip(grads, vars_list))
            return self.critic_optimizer.apply_gradients(grads)

        qf1_train_op = minimize(self.td_loss1_t, self.qf1.get_params(), self.max_grad_norm)
        qf2_train_op = minimize(self.td_loss2_t, self.qf2.get_params(), self.max_grad_norm)
        self.critic_train_ops.append(qf1_train_op)
        self.diagnosis_op.append(self.td_loss1_t)
        if self.two_qf:
            self.critic_train_ops.append(qf2_train_op)
            self.diagnosis_op.append(self.td_loss2_t)

    def _init_actor_update(self):

        acts_n, raw_acts_n, logps_n, dists_n = [], [], [], []
        old_acts_n, old_raw_acts_n, old_logps_n, old_dists_n = [], [], [], []
        for i in range(self.nagent):
            acts, raw_acts, logps, dists = self.pi_n[i].forward(self.obs_ph_n[i])
            acts_n.append(acts)
            raw_acts_n.append(raw_acts)
            logps_n.append(logps)
            dists_n.append(dists)

            old_acts, old_raw_acts, old_logps, old_dists = self.oldpi_n[i].forward(self.obs_ph_n[i])
            old_acts_n.append(old_acts)
            old_raw_acts_n.append(old_raw_acts)
            old_logps_n.append(old_logps)
            old_dists_n.append(old_dists)

        logpas_joint = tf.reduce_sum(tf.concat([logps_n[i][..., None] for i in range(self.nagent)], axis=-1), axis=-1)
        approx_entropy = -tf.reduce_mean(logpas_joint)
        self.diagnosis_op.append(approx_entropy)
        self.diagnosis_op.append(self.alpha)

        self.alpha_train_op = []
        if self.auto_alpha:
            alpha_loss = tf.reduce_mean(-self.alpha * (logpas_joint + self.target_entropy))
            self.alpha_train_op.append(self.alpha_optimizer.minimize(alpha_loss, var_list=[self.log_alpha]))

        # policy train op
        self.mean_kl_n = []
        self.beta_train_op_n = []
        for i in range(self.nagent):
            old_acts_n_copy = old_acts_n[:]
            old_acts_n_copy[i] = acts_n[i]
            q1 = self.qf1.forward(self.states_ph, old_acts_n_copy)
            q2 = self.qf2.forward(self.states_ph, old_acts_n_copy)
            if self.two_qf:
                min_q = tf.minimum(q1, q2)
            else:
                min_q = q1

            pg_loss = tf.reduce_mean(self.alpha * logps_n[i] - min_q)  # minimize negative target
            mean_kl = tf.reduce_mean(old_dists_n[i].pd.kl(dists_n[i].pd))
            if self.trpo:
                pg_loss += self.beta_n[i] * mean_kl
                beta_loss = -self.beta_n[i] * (mean_kl - self.delta_n[i])
                beta_train_op = self.beta_optimizer.minimize(beta_loss, var_list=[self.log_beta_n[i]])
                self.beta_train_op_n.append(beta_train_op)
            pg_grads = tf.gradients(pg_loss, self.pi_n[i].get_params())
            if self.max_grad_norm is not None:
                pg_grads, _grad_norm = tf.clip_by_global_norm(pg_grads, self.max_grad_norm)  # clip gradients
            pg_grads = list(zip(pg_grads, self.pi_n[i].get_params()))
            pg_train_op = self.pi_optimizer.apply_gradients(pg_grads)

            self.pi_train_op_n.append(pg_train_op)
            self.mean_kl_n.append(mean_kl)

    def _init_target_ops(self):
        # actor
        actor_tau = tf.placeholder(dtype=tf.float32, shape=[], name='actor_tau')
        source_params = list(np.ravel([self.pi_n[i].get_params() for i in range(self.nagent)]))
        target_params = list(np.ravel([self.oldpi_n[i].get_params() for i in range(self.nagent)]))
        actor_target_ops = [tf.assign(target, (1 - actor_tau) * target + actor_tau * source)
                            for target, source in zip(target_params, source_params)]

        # critic
        critic_tau = tf.placeholder(dtype=tf.float32, shape=[], name='critic_tau')
        if self.two_qf:
            source_params = self.qf1.get_params() + self.qf2.get_params()
            target_params = self.qf1_target_params + self.qf2_target_params
        else:
            source_params = self.qf1.get_params()
            target_params = self.qf1_target_params

        critic_target_ops = [tf.assign(target, (1 - critic_tau) * target + critic_tau * source)
                             for target, source in zip(target_params, source_params)]
        self.update_target = U.function([actor_tau, critic_tau], [],
                                        updates=[actor_target_ops, critic_target_ops])

    def train(self, load_path=None):
        if load_path:
            self.load(load_path)
        start_time = time.time()
        for epoch in range(self.n_epochs):
            t_start = time.time()
            mean_kl_n = []
            for t in range(self.epoch_length):
                self.runner.sample()
                if not self.runner.batch_ready():
                    continue
                iteration = t + epoch * self.epoch_length
                for i in range(self.n_train_repeat):
                    mean_kl_n = self.do_training(iteration)
            self._evaluate()
            if len(mean_kl_n) != 0:
                logger.logkv("mean_kl", np.round(mean_kl_n, 6))
            t_now = time.time()
            logger.logkv('epoch', epoch)
            logger.logkv('fps', self.epoch_length / (t_now - t_start))
            logger.logkv('time_elapsed', t_now - start_time)
            self.runner.log_diagnostics()
            logger.dumpkvs()
            if self.save_interval and (epoch % self.save_interval == 0 or epoch == 1) and logger.get_dir():
                checkdir = osp.join(logger.get_dir(), 'checkpoints')
                os.makedirs(checkdir, exist_ok=True)
                savepath = osp.join(checkdir, '%.5i' % epoch)
                print('Saving to', savepath)
                self.save(savepath)

    def _evaluate(self):
        batch = self.runner.random_batch()
        diagnosis = self.sess.run(self.diagnosis_op + self.beta_n, self.get_feed_dict(batch))
        rewards_n = [rollout(self.eval_env, self.get_action) for _ in range(self.eval_n_episodes)]
        total_returns = [rewards.sum() for rewards in rewards_n]
        episode_lengths = [len(rewards) for rewards in rewards_n]

        logger.logkv("agent", np.array([agent for agent in range(self.nagent)]))
        for lossname, val in zip(self.loss_names, diagnosis[:-self.nagent]):
            logger.logkv(lossname, val)
        if self.trpo:
            logger.logkv("beta", np.round(diagnosis[-self.nagent:], 4))
        logger.logkv('return-average', np.mean(total_returns))
        logger.logkv('episode-length-avg', np.mean(episode_lengths))

    def do_training(self, iteration):
        mean_kl_n = []
        if iteration % self.update_interval == 0:
            batch = self.runner.random_batch()
            feed_dict = self.get_feed_dict(batch)
            # lr setting
            frac = 1 - iteration / self.total_time_steps
            pi_lr_now = self.pi_lr
            critic_lr_now = self.critic_lr
            if self.pi_lr_decay:
                pi_lr_now *= frac
            if self.critic_lr_decay:
                critic_lr_now *= frac
            feed_dict[self.pi_LR] = pi_lr_now
            feed_dict[self.critic_LR] = critic_lr_now

            run_op = self.pi_train_op_n + self.critic_train_ops + self.alpha_train_op
            self.sess.run(run_op, feed_dict)
            mean_kl_n = self.sess.run(self.beta_train_op_n + self.mean_kl_n, feed_dict)[-self.nagent:]

        # update target
        if iteration % self.target_update_interval == 0:
            self.update_target(self.actor_tau, self.critic_tau)

        return mean_kl_n

    def get_feed_dict(self, batch):
        obs_n = batch['obs_n']
        next_obs_n = batch['next_obs_n']
        acts_n = batch['acts_n']
        feed_dict = {self.states_ph: batch['states'],
                     self.next_states_ph: batch['next_states'],
                     self.rewards_ph: batch['rewards'],
                     self.masks_ph: batch['masks']}
        feed_dict.update({self.obs_ph_n[i]: obs_n[i] for i in range(self.nagent)})
        feed_dict.update({self.next_obs_ph_n[i]: next_obs_n[i] for i in range(self.nagent)})
        feed_dict.update({self.acts_ph_n[i]: acts_n[i] for i in range(self.nagent)})
        return feed_dict

    def get_actions(self, obs_n, is_deterministic=False):
        feed_dict = {self.oldpi_n[i].obs_ph: obs_n[i] for i in range(self.nagent)}
        if is_deterministic:
            run_op = [self.oldpi_n[i].dist.mean for i in range(self.nagent)]
            acts_n = self.sess.run(run_op, feed_dict)
            if self.squash:
                acts_n = [np.tanh(acts) for acts in acts_n]
        else:
            run_op = [self.oldpi_n[i].actions for i in range(self.nagent)]
            acts_n = self.sess.run(run_op, feed_dict)
        return acts_n

    def get_action(self, ob_n, is_deterministic=False):
        ob_n = [ob[None] for ob in ob_n]
        act_n = self.get_actions(ob_n, is_deterministic)
        return [act[0] for act in act_n]

    @staticmethod
    def save(save_path):
        params = tf.trainable_variables()
        ps = tf.get_default_session().run(params)
        joblib.dump(ps, save_path)

    @staticmethod
    def load(load_path):
        if load_path is not None:
            loaded_params = joblib.load(load_path)
            params = tf.trainable_variables()
            restores = []
            for p, loaded_p in zip(params, loaded_params):
                restores.append(p.assign(loaded_p))
            tf.get_default_session().run(restores)
