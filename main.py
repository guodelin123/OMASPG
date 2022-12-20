import os
import yaml
import argparse
import tensorflow as tf

from pprint import pprint
from core.model import Model
from test_envs.env_maker import MaMujocoEnv
from common import utils, logger
from common.utils import str2bool, int_or_None, float_or_None

def parse_args():
    parser = argparse.ArgumentParser("Reinforcement Learning experiments for multiagent environments")
    # Environment args,
    parser.add_argument("--scenario", type=str, default="HalfCheetah-v2", help="scenario name")
    parser.add_argument("--agent-conf", type=str, default='2x3',
                        help="See 'https://github.com/schroederdewitt/multiagent_mujoco' for introduction")
    parser.add_argument("--agent-obsk", type=int_or_None, default=None,
                        help="See 'https://github.com/schroederdewitt/multiagent_mujoco' for introduction")
    parser.add_argument("--episode-limit", type=int, default=1000,
                        help="200 for HumanoidStandup and 1000 for the others")

    # training parameters, fixed
    parser.add_argument("--n-epochs", type=int, default=int(4001), help="Total learning epoches")
    parser.add_argument("--epoch-length", type=int, default=int(1000), help="Length of each epoch")
    parser.add_argument("--n-train-repeat", type=int, default=1, help="Number of times per training step")
    parser.add_argument("--update-interval", type=int, default=1, help="Update interval")
    parser.add_argument("--target-update-interval", type=int, default=1, help="Update interval of targets")
    parser.add_argument("--two_qf", type=str2bool, default=True, help="two q function")
    parser.add_argument("--pi-lr", type=float, default=3e-4, help="learning rate for policies")
    parser.add_argument("--critic-lr", type=float, default=3e-4, help="learning rate for critics")
    parser.add_argument("--pi-lr-decay", type=str2bool, default=False, help="If True, pi_lr is annealing")
    parser.add_argument("--critic-lr-decay", type=str2bool, default=False, help="If True, critic_lr is annealing")
    parser.add_argument("--alpha-lr", type=float, default=3e-4, help="learning rate for the entropy coefficient")
    parser.add_argument("--critic-tau", type=float, default=0.005, help="moving average parameter of the target critic")
    parser.add_argument("--actor-tau", type=float, default=0.005, help="moving average parameter of the policy")
    parser.add_argument("--gamma", type=float, default=0.99, help="discount factor")
    parser.add_argument("--batch-size", type=int, default=256, help="batch size of training data")
    parser.add_argument("--squash", type=str2bool, default=True, help="If true, squashing actions to (-1,1)")
    parser.add_argument("--max-grad-norm", type=float_or_None, default=5., help='max norm of gradients')
    parser.add_argument("--huber-delta", type=float_or_None, default=10., help="huber delta if use huber loss")

    parser.add_argument("--trpo", type=str2bool, default=True, help="If True, the KL-penalty will be used")
    parser.add_argument("--beta-optimizer", type=str, default='adam', help="optimizer of beta")
    parser.add_argument("--beta-lr", type=float, default=1e-4, help="Learning rate for the KL-penalty coefficient")
    parser.add_argument("--delta-base", type=float, default=0.005, help="The disired KL per action dim")
    parser.add_argument("--r-scale", type=float, default=1., help="reward scale in training")

    # may be different in different tasks
    parser.add_argument("--auto-alpha", type=str2bool, default=True,
                        help="If True, alpha will be tuned automatically, otherwise fixed")
    parser.add_argument("--alpha", type=float, default=0.1, help="If auto_alpha is True, this arg will be muted")

    # buffer set, fixed
    parser.add_argument("--max-buffer-size", type=int, default=int(1e6), help="buffer size")
    parser.add_argument("--min-buffer-size", type=int, default=1000, help="min buffer size for training")

    parser.add_argument("--n-node", type=int, default=256, help="number of units in the mlp")
    parser.add_argument("--seed", type=int_or_None, default=None, help="random seed")
    parser.add_argument("--eval-n-episodes", type=int, default=1)

    # Checkpointing
    parser.add_argument("--save_interval", type=int, default=500,
                        help="save model once every time this many episodes are completed")

    return parser.parse_args()


if __name__ == '__main__':

    args = parse_args()
    pprint(args.__dict__)
    utils.set_global_seed(args.seed)

    save_dir = 'results/' + args.scenario + '/' + utils.get_token(True)
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    logger.configure(save_dir)
    with open(save_dir + '/args_dict.yaml', 'w') as f:
        yaml.dump(args.__dict__, f)123

    config = tf.ConfigProto(allow_soft_placement=True)
    tf.GPUOptions(per_process_gpu_memory_fraction=0.1)
    config.gpu_options.allow_growth = True
    tf.InteractiveSession(config=config)

    # make environment
    env = MaMujocoEnv(scenario=args.scenario,
                      agent_conf=args.agent_conf,
                      agent_obsk=args.agent_obsk,
                      episode_limit=args.episode_limit)

    # episode_limit eval env is fixed as 1000
    eval_env = MaMujocoEnv(scenario=args.scenario,
                           agent_conf=args.agent_conf,
                           agent_obsk=args.agent_obsk,
                           episode_limit=1000)

    algorithm = Model(env, eval_env, args)
    algorithm.train()
