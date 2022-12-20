
# OMASPG: An Off-Policy Multi-Agent Stochastic Policy Gradient Algorithm for Cooperative Continuous Control

This is the code for implementing the OMASPG algorithm,
It is configured to be run in conjunction with environments from the
[Multi-Agent MuJoCo (MAMuJoCo)](https://github.com/schroederdewitt/multiagent_mujoco).

## Known dependencies
Python (3.6.13), OpenAI gym (0.19.0), tensorflow (1.12.0), numpy (1.19.5), mujoco-py (1.50.1.0).
- Download and install the [MAMuJoCo](https://github.com/schroederdewitt/multiagent_mujoco) by following its `README`.
- Different from the guidance in [MAMuJoCo](https://github.com/schroederdewitt/multiagent_mujoco), we use Mujoco 1.50 instead of 2.1.

## Training
1. `cd` into the root directory.
2. Take HalfCheetah (2x3) for example, run the following command: \
`python main.py --scenario=HalfCheetah-v2 --agent-conf=2x3`

## Testing
We have included our well-trained policies in `demos/`.\
Take HumanoidStandup for example, you can run the command:\
 `python test.py --load-dir=demos/HumanoidStandup --render=True`
