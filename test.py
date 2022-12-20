import os
import yaml
import numpy as np
import tensorflow as tf
import argparse
from core.runner import rollout
from common.utils import str2bool
from core.model import Model
from test_envs.env_maker import MaMujocoEnv
from types import SimpleNamespace as SN

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"


def test_model(load_dir, deter=True, n_test=10, sleep_time=0.01, render=True):
    with open(load_dir + '/args_dict.yaml', "r") as file:
        config_dict = yaml.load(file)
    args = SN(**config_dict)
    tf.InteractiveSession()
    env = MaMujocoEnv(scenario=args.scenario,
                      agent_conf=args.agent_conf,
                      agent_obsk=args.agent_obsk,
                      episode_limit=1000)

    model = Model(env, env, args)
    model.sess.run(tf.global_variables_initializer())

    check_point = load_dir + '/checkpoints/'
    points = os.listdir(check_point)
    points.sort()
    dir = check_point + points[-1]
    print('load_dir:', dir)
    model.load(dir)
    average_rewards = []
    for _ in range(n_test):
        rewards = rollout(env, model.get_action, deter, sleep_time, render)
        average_rewards.append(np.sum(rewards))
    mean_r = np.mean(average_rewards)
    print('average_return:', mean_r)
    return mean_r


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--load-dir", type=str, default="demos/HalfCheetah")
    parser.add_argument("--render", type=str2bool, default=False)
    return parser.parse_args()


if __name__ == '__main__':
    arg = parse_args()
    test_model(arg.load_dir, deter=True, n_test=32, sleep_time=0.01, render=arg.render)
