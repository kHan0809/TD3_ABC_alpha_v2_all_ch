import numpy as np
import torch
import gym
import argparse
import os
import d4rl

import utils
import TD3_BC
import matplotlib.pyplot as plt

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
if __name__ == "__main__":


	div_std = 1
	algo = "TD3_IABC/"
	x=np.linspace(10000,1e6,100)

	p_dir = "./results/"+algo
	ext = ".npy"
	total_sum = 0
	for idx,env in enumerate(envs):
		data = []
		for file_name in os.listdir(p_dir):
			if env in file_name:
				data.append(np.load(p_dir+file_name))
			else:
				pass
		mean = np.array(data).mean(axis=0)[0::2]
		std  = np.array(data).std(axis=0)[0::2]
		total_sum += mean[-1]

		plt.subplot(3, 3, idx + 1,)
		if idx == 8:
			plt.plot(x, mean, linewidth=0.5, label="TD3+IABC")
		else:
			plt.plot(x, mean, linewidth=0.5)
		plt.fill_between(x,mean-std/div_std, mean+std/div_std, alpha=0.2)
		plt.title(env,fontsize=10)
		plt.xlabel("Training step",fontsize=10)
		if idx == 0:
			plt.ylabel("Normalized Returns",fontsize=10)
		plt.grid()
		plt.subplots_adjust(left=0.2, bottom=0.1, right=1.2, top=0.9, wspace=0.3, hspace=0.4)

	algo = "TD3_BC/"

	p_dir = "./results/"+algo
	ext = ".npy"
	total_sum = 0
	for idx,env in enumerate(envs):
		data = []
		for file_name in os.listdir(p_dir):
			if env in file_name:
				data.append(np.load(p_dir+file_name))
			else:
				pass
		mean = np.array(data).mean(axis=0)[0::2]
		std  = np.array(data).std(axis=0)[0::2]
		total_sum += mean[-1]

		plt.subplot(3, 3, idx + 1)
		if idx==8:
			plt.plot(x,mean,linewidth=0.5,label="TD3+BC")
		else:
			plt.plot(x, mean, linewidth=0.5)
		plt.fill_between(x,mean-std/div_std, mean+std/div_std, alpha=0.2)
		plt.title(env,fontsize=10)
		plt.xlabel("Training step",fontsize=10)
		if idx == 0:
			plt.ylabel("Normalized Returns",fontsize=10)
		plt.subplots_adjust(left=0.125, bottom=0.1, right=0.9, top=0.9, wspace=0.3, hspace=0.4)

	algo = "BC/"

	p_dir = "./results/"+algo
	ext = ".npy"
	total_sum = 0
	for idx,env in enumerate(envs):
		data = []
		for file_name in os.listdir(p_dir):
			if env in file_name:
				data.append(np.load(p_dir+file_name))
			else:
				pass
		mean = np.array(data).mean(axis=0)[0::2]
		std  = np.array(data).std(axis=0)[0::2]
		total_sum += mean[-1]

		plt.subplot(3, 3, idx + 1)
		if idx==8:
			plt.plot(x,mean,linewidth=0.5,label="BC")
		else:
			plt.plot(x, mean, linewidth=0.5)
		plt.fill_between(x,mean-std/div_std, mean+std/div_std, alpha=0.2)
		plt.title(env,fontsize=10)
		plt.xlabel("Training step",fontsize=10)
		if idx == 0:
			plt.ylabel("Normalized Returns",fontsize=10)
		plt.subplots_adjust(left=0.125, bottom=0.1, right=0.9, top=0.9, wspace=0.3, hspace=0.4)

	algo = "AWR/"

	p_dir = "./results/"+algo
	ext = ".npy"
	total_sum = 0
	for idx,env in enumerate(envs):
		data = []
		for file_name in os.listdir(p_dir):
			if env in file_name:
				data.append(np.load(p_dir+file_name))
			else:
				pass
		mean = np.array(data).mean(axis=0)[0::2]
		std  = np.array(data).std(axis=0)[0::2]
		total_sum += mean[-1]

		plt.subplot(3, 3, idx + 1)
		if idx==8:
			plt.plot(x,mean,linewidth=0.5,label="AWR")
		else:
			plt.plot(x, mean, linewidth=0.5)
		plt.fill_between(x,mean-std/div_std, mean+std/div_std, alpha=0.2)
		plt.title(env,fontsize=10)
		plt.xlabel("Training step",fontsize=10)
		if idx == 0:
			plt.ylabel("Normalized Returns",fontsize=10)
		plt.subplots_adjust(left=0.125, bottom=0.1, right=0.9, top=0.9, wspace=0.3, hspace=0.4)

	algo = "TD3_IABCwDprime/"

	p_dir = "./results/"+algo
	ext = ".npy"
	total_sum = 0
	for idx,env in enumerate(envs):
		data = []
		for file_name in os.listdir(p_dir):
			if env in file_name:
				data.append(np.load(p_dir+file_name))
			else:
				pass
		mean = np.array(data).mean(axis=0)[0::2]
		std  = np.array(data).std(axis=0)[0::2]
		total_sum += mean[-1]

		plt.subplot(3, 3, idx + 1)
		if idx==8:
			plt.plot(x,mean,linewidth=0.5,label="TD3+IABC with D'")
		else:
			plt.plot(x, mean, linewidth=0.5)
		plt.fill_between(x,mean-std/div_std, mean+std/div_std, alpha=0.2)
		plt.title(env,fontsize=10)
		plt.xlabel("Training step",fontsize=10)
		if idx == 0:
			plt.ylabel("Normalized Returns",fontsize=10)
		plt.subplots_adjust(left=0.125, bottom=0.1, right=0.9, top=0.9, wspace=0.3, hspace=0.4)

	algo = "only_TD3_IABC/"

	p_dir = "./results/"+algo
	ext = ".npy"
	total_sum = 0
	for idx,env in enumerate(envs):
		data = []
		for file_name in os.listdir(p_dir):
			if env in file_name:
				data.append(np.load(p_dir+file_name))
			else:
				pass
		mean = np.array(data).mean(axis=0)[0::2]
		std  = np.array(data).std(axis=0)[0::2]
		total_sum += mean[-1]

		plt.subplot(3, 3, idx + 1)
		if idx==8:
			plt.plot(x,mean,linewidth=0.5,label="only TD3+IABC")
		else:
			plt.plot(x, mean, linewidth=0.5)
		plt.fill_between(x,mean-std/div_std, mean+std/div_std, alpha=0.2)
		plt.title(env,fontsize=10)
		plt.xlabel("Training step",fontsize=10)
		if idx == 0:
			plt.ylabel("Normalized Returns",fontsize=10)
		plt.subplots_adjust(left=0.125, bottom=0.1, right=0.9, top=0.9, wspace=0.3, hspace=0.4)


	plt.legend(loc='lower left')
	plt.show()