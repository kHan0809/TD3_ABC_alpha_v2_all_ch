import numpy as np
import torch
import copy

class ReplayBuffer(object):
	def __init__(self, state_dim, action_dim, max_size=int(2e6)):
		self.max_size = max_size
		self.ptr = 0
		self.size = 0

		self.state = np.zeros((max_size, state_dim))
		self.action = np.zeros((max_size, action_dim))
		self.next_state = np.zeros((max_size, state_dim))
		self.reward = np.zeros((max_size, 1))
		self.not_done = np.zeros((max_size, 1))

		self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


	def addReturn(self,dataset):
		Return_buffer = []
		tmp_return = 0
		for i in reversed(range(self.size)):
			if dataset['terminals'][i] or dataset["timeouts"][i] or i == (self.size - 1):
				tmp_return = dataset["rewards"][i]
				Return_buffer.append(tmp_return)
			else:
				tmp_return = dataset["rewards"][i] + 0.99 * tmp_return
				Return_buffer.append(tmp_return)

		Return_buffer = copy.deepcopy(np.flip(np.array(Return_buffer, dtype=np.float32)))
		self.Return = Return_buffer.reshape(-1,1)

	def add(self, state, action, next_state, reward, done):
		self.state[self.ptr] = state
		self.action[self.ptr] = action
		self.next_state[self.ptr] = next_state
		self.reward[self.ptr] = reward
		self.not_done[self.ptr] = 1. - done

		self.ptr = (self.ptr + 1) % self.max_size
		self.size = min(self.size + 1, self.max_size)


	def sample(self, batch_size):
		ind = np.random.randint(0, self.size, size=batch_size)

		return (
			torch.FloatTensor(self.state[ind]).to(self.device),
			torch.FloatTensor(self.action[ind]).to(self.device),
			torch.FloatTensor(self.next_state[ind]).to(self.device),
			torch.FloatTensor(self.reward[ind]).to(self.device),
			torch.FloatTensor(self.not_done[ind]).to(self.device),
			torch.FloatTensor(self.Return[ind]).to(self.device),
		)


	def convert_D4RL(self, dataset):
		self.state = dataset['observations']
		self.action = dataset['actions']
		self.next_state = dataset['next_observations']
		self.reward = dataset['rewards'].reshape(-1,1)
		self.not_done = 1. - dataset['terminals'].reshape(-1,1)
		self.timeout = dataset['timeouts'].reshape(-1,1)
		self.size = self.state.shape[0]
		self.addReturn(dataset)


	def normalize_states(self, eps = 1e-3):
		mean = self.state.mean(0,keepdims=True)
		std = self.state.std(0,keepdims=True) + eps
		self.state = (self.state - mean)/std
		self.next_state = (self.next_state - mean)/std
		return mean, std

	def check_epi_rank(self):
		terminal = 1. - self.not_done
		temp = terminal + self.timeout
		self.idx = np.where(temp  > 0)[0]
		self.idx = np.concatenate((np.array([0]), self.idx[:-1] + 1))
		self.epi_num=self.idx.shape[0]
		print('idx :',self.idx, "epi_num",self.epi_num)
		rank = self.Return[self.idx,0].argsort()
		tmp = np.array([])
		for i in range(self.epi_num-1):
			tmp = np.append(tmp, np.ones((self.idx[i + 1] - self.idx[i])) * rank[i])
		tmp = np.append(tmp, np.ones((self.Return.shape[0] - self.idx[-1])) * rank[-1])
		self.rank = tmp
		print(self.rank)
		return self.epi_num

	def get_epi(self, i):
		return (
			torch.FloatTensor(self.state[self.idx[i]:self.idx[i+1]]).to(self.device),
			torch.FloatTensor(self.action[self.idx[i]:self.idx[i+1]]).to(self.device),
			torch.FloatTensor(self.next_state[self.idx[i]:self.idx[i+1]]).to(self.device),
			torch.FloatTensor(self.reward[self.idx[i]:self.idx[i+1]]).to(self.device),
			torch.FloatTensor(self.not_done[self.idx[i]:self.idx[i+1]]).to(self.device),
			torch.FloatTensor(self.Return[self.idx[i]:self.idx[i+1]]).to(self.device),
			torch.FloatTensor(self.rank[self.idx[i]:self.idx[i+1]]).to(self.device),
		)