import numpy as np
import torch
import gym
import argparse
import os
import d4rl

import utils
import matplotlib.pyplot as plt
if __name__ == "__main__":

	envs = [
		"halfcheetah-medium-v2",
		"hopper-medium-v2",
		"walker2d-medium-v2",
		"halfcheetah-medium-expert-v2",
		"hopper-medium-expert-v2",
		"walker2d-medium-expert-v2",
		"halfcheetah-medium-replay-v2",
		"hopper-medium-replay-v2",
		"walker2d-medium-replay-v2",
	]
	algo = "TD3_ABC/"
	x = np.linspace(5000, 1e6, 200)

	p_dir = "./results/" + algo
	ext = ".npy"
	total_sum = 0
	for idx, env in enumerate(envs):
		data = []
		for file_name in os.listdir(p_dir):
			if env in file_name:
				data.append(np.load(p_dir + file_name))
			else:
				pass
		mean = np.array(data).mean(axis=0)
		std  = np.array(data).std(axis=0)
		total_sum += mean[-1]

		plt.subplot(3, 5, idx + 1)
		plt.plot(mean)
		plt.fill_between(range(1,201),mean-std, mean+std, alpha=0.5)
		plt.title(env)
		print("Env : %30s"% env, "mean : %5.3f" % mean[-1], "std : %5.3f" % std[-1])
	print("Total : ", total_sum)
	plt.show()




	# 	try:
	#
	# 		data = np.load(p_dir + file_name + ext)
	# 		plt.subplot(3,5,idx+1)
	# 		plt.plot(data)
	# 		plt.title(env)
	# 		plt.xlabel("time steps")
	# 	except:
	# 		pass
	# plt.show()

# "halfcheetah-random-v2"
# "hopper-random-v2"
# "walker2d-random-v2"
# "halfcheetah-medium-v2"
# "hopper-medium-v2"
# "walker2d-medium-v2"
# "halfcheetah-medium-expert-v2"