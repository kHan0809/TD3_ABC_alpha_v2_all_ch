import numpy as np
import torch
import gym
import argparse
import os
import d4rl

import utils
import TD3_BC
import matplotlib.pyplot as plt

# Runs policy for X episodes and returns D4RL score
# A fixed seed is used for the eval environment


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    # Experiment
    parser.add_argument("--policy", default="TD3_BC")  # Policy name
    parser.add_argument("--env", default="hopper-medium-expert-v2")  # OpenAI gym environment name
    parser.add_argument("--seed", default=1, type=int)  # Sets Gym, PyTorch and Numpy seeds
    parser.add_argument("--eval_freq", default=5e3, type=int)  # How often (time steps) we evaluate
    parser.add_argument("--max_timesteps", default=1e6, type=int)  # Max time steps to run environment
    parser.add_argument("--max_timesteps_Q", default=3e4, type=int)  # Max time steps to run environment
    parser.add_argument("--save_model", action="store_true")  # Save model and optimizer parameters
    parser.add_argument("--load_model", default="")  # Model load file name, "" doesn't load, "default" uses file_name
    # TD3
    parser.add_argument("--expl_noise", default=0.1)  # Std of Gaussian exploration noise
    parser.add_argument("--batch_size", default=256, type=int)  # Batch size for both actor and critic
    parser.add_argument("--discount", default=0.99)  # Discount factor
    parser.add_argument("--tau", default=0.005)  # Target network update rate
    parser.add_argument("--policy_noise", default=0.2)  # Noise added to target policy during critic update
    parser.add_argument("--noise_clip", default=0.5)  # Range to clip target policy noise
    parser.add_argument("--policy_freq", default=2, type=int)  # Frequency of delayed policy updates
    parser.add_argument("--clutch", default=9e5, type=int)  # Frequency of delayed policy updates
    # TD3 + BC
    parser.add_argument("--alpha", default=2.5)
    parser.add_argument("--normalize", default=True)
    args = parser.parse_args()

    file_name = f"{args.policy}_{args.env}_{args.seed}"
    print("---------------------------------------")
    print(f"Policy: {args.policy}, Env: {args.env}, Seed: {args.seed}")
    print("---------------------------------------")

    if not os.path.exists("./results"):
        os.makedirs("./results")

    if args.save_model and not os.path.exists("./models"):
        os.makedirs("./models")

    env = gym.make(args.env)

    # Set seeds
    env.seed(args.seed)
    env.action_space.seed(args.seed)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]
    max_action = float(env.action_space.high[0])

    kwargs = {
        "state_dim": state_dim,
        "action_dim": action_dim,
        "max_action": max_action,
        "discount": args.discount,
        "tau": args.tau,
        # TD3
        "policy_noise": args.policy_noise * max_action,
        "noise_clip": args.noise_clip * max_action,
        "policy_freq": args.policy_freq,
        # TD3 + BC
        "alpha": args.alpha
    }

    # Initialize policy
    policy = TD3_BC.TD3_BC(**kwargs)

    if args.load_model != "":
        policy_file = file_name if args.load_model == "default" else args.load_model
        policy.load(f"./models/{policy_file}")

    policy.value.load_state_dict(torch.load("./model/"+args.env+".pt"))

    replay_buffer = utils.ReplayBuffer(state_dim, action_dim)
    replay_buffer.convert_D4RL(env.get_dataset())
    epi_len = replay_buffer.check_epi_rank()
    print(epi_len)


    if args.normalize:
        mean, std = replay_buffer.normalize_states()
    else:
        mean, std = 0, 1
    ranks, weights = [], []
    for i in range(epi_len-1):
        weight,rank=policy.test_epi_rv(replay_buffer,i)
        weights.append(weight)
        ranks.append(rank)
    weights = np.array(weights)
    ranks   = np.array(ranks)
    idx=np.argsort(ranks)
    ranks = ranks[idx]
    weights = weights[idx]
    plt.plot(ranks[:2175], weights[:2175], 'o', color='skyblue',label="medium")
    plt.plot(ranks[2175:], weights[2175:], 'o', color='salmon',label="expert")

    x = np.array([i * 25 for i in range(1,epi_len // 25 + 1)])
    interval_weight = []
    for i in range((epi_len//25)-1):
        interval_weight.append(weights[x[i]:x[i+1]].mean())
    interval_weight.append(weights[x[i+1]:x[i + 1]+25].mean())
    interval_weight = np.array(interval_weight)
    plt.plot(x, interval_weight, 'o-',color='orange')
    plt.title(args.env[:-3])
    plt.xlabel("Episode Return Rank")
    plt.ylabel("Mean of Exponential Advantage Weight")
    # plt.fill_between(np.array([0, 2175]),np.array([0,0]),np.array([10,10]),color='g',alpha=0.3,)
    # plt.fill_between(np.array([2175, 3213]), np.array([0, 0]), np.array([10, 10]), color='m', alpha=0.3)
    plt.legend()
    plt.grid()
    plt.show()






