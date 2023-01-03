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
import copy

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    # Experiment
    parser.add_argument("--policy", default="TD3_BC")  # Policy name
    parser.add_argument("--env", default="halfcheetah-medium-expert-v2")  # OpenAI gym environment name
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

    env = gym.make("halfcheetah-medium-v2")
    env1 = gym.make("halfcheetah-expert-v2")

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

    policy.value.load_state_dict(torch.load("./model/"+args.env+".pt"))
    # policy.value.load_state_dict(torch.load("./model/hopper-medium-expert-append-v2.pt"))

    replay_buffer = utils.ReplayBuffer(state_dim, action_dim)
    replay_buffer.convert_D4RL(env.get_dataset())
    epi_len = replay_buffer.check_epi_rank()

    replay_buffer1 = utils.ReplayBuffer(state_dim, action_dim)
    replay_buffer1.convert_D4RL(env1.get_dataset())
    epi_len1 = replay_buffer1.check_epi_rank()



    if args.normalize:
        mean, std = replay_buffer.normalize_states()
        mean, std = replay_buffer1.normalize_states()
    else:
        mean, std = 0, 1

    fig, ax1 = plt.subplots()
    ax2 = ax1.twinx()
    print("==================================medium")
    mReturns, mweights, mzero_nums = [], [], []
    for i in range(epi_len-1):
        mweight,mReturn,mzero_num=policy.test_epi_rv(replay_buffer, i)
        mweights.append(mweight)
        mReturns.append(mReturn)
        mzero_nums.append(mzero_num)
    mweights = np.array(mweights)
    mReturns   = np.array(mReturns)
    mzero_nums = np.array(mzero_nums)
    print("==================================expert")
    eReturns, eweights, ezero_nums = [], [], []
    for i in range(epi_len1-1):
        eweight, eReturn, ezero_num = policy.test_epi_rv(replay_buffer1, i)
        eweights.append(eweight)
        eReturns.append(eReturn)
        ezero_nums.append(ezero_num)
    eweights = np.array(eweights)
    eReturns = np.array(eReturns)
    ezero_nums = np.array(ezero_nums)

    all_return = copy.deepcopy(np.concatenate((mReturns,eReturns),axis=0))
    idx=np.argsort(all_return.reshape(-1))
    ax1.plot(idx[:999], mweights, 'o', color='skyblue',label="medium")
    ax1.plot(idx[999:], eweights, 'o', color='salmon',label="expert")

    weights = np.concatenate((mweights, eweights), axis=0)
    weights =  weights[idx]
    zero_nums = np.concatenate((mzero_nums, ezero_nums), axis=0)
    zero_nums = zero_nums[idx]

    x = np.array([i * 25 for i in range(1,(epi_len+epi_len1)//25 + 1)])
    interval_weight = []
    for i in range((epi_len+epi_len1)//25-1):
        interval_weight.append(weights[x[i]:x[i+1]].mean())
    interval_weight.append(weights[x[i+1]:x[i + 1]+25].mean())
    interval_weight = np.array(interval_weight)
    ax1.plot(x, interval_weight, 'o-', color='orange')

    interval_zero = []
    for i in range((epi_len+epi_len1)//25-1):
        # print(zero_nums[x[i]:x[i+1]].shape,x[i],x[i+1])
        interval_zero.append(zero_nums[x[i]:x[i+1]].mean())
    interval_zero.append(zero_nums[x[i+1]:x[i + 1]+25].mean())
    interval_zero = np.array(interval_zero)
    print(x.shape,interval_zero.shape)
    ax2.plot(x, interval_zero*5,'o-',color='grey')



    plt.title(args.env[:-3])
    ax1.set_xlabel("Episode Return Rank")
    ax1.set_ylabel("Mean of Exponential Advantage Weight")
    ax2.set_ylabel("Normalized Number of Negative Advantage")
    ax1.grid()
    ax1.legend(loc='upper left')
    plt.show()






