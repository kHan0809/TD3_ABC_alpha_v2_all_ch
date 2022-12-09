import copy
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class Actor(nn.Module):
	def __init__(self, state_dim, action_dim, max_action):
		super(Actor, self).__init__()

		self.l1 = nn.Linear(state_dim, 256)
		self.l2 = nn.Linear(256, 256)
		self.l3 = nn.Linear(256, action_dim*2)
		
		self.max_action = max_action
		self.log_std_min = -20
		self.log_std_max = 2
		

	def forward(self, state, eval=False):
		a = F.relu(self.l1(state))
		a = F.relu(self.l2(a))
		mean, log_std = self.l3(a).chunk(2, dim=-1)
		log_std = torch.clamp(log_std, self.log_std_min, self.log_std_max)
		std = log_std.exp()
		dist = torch.distributions.Normal(mean, std, validate_args=True)
		if eval:
			return self.max_action * torch.tanh(mean)
		else:
			sample_action = dist.sample()
			tanh_sample = torch.tanh(sample_action)
			log_prob = dist.log_prob(sample_action)
			log_pi = log_prob - torch.log(1 - tanh_sample.pow(2) + 1e-6).sum(dim=1, keepdim=True)
			return self.max_action * tanh_sample, log_pi

	def get_log_prob(self, state, action_batch):
		a = F.relu(self.l1(state))
		a = F.relu(self.l2(a))
		mean, log_std = self.l3(a).chunk(2, dim=-1)
		log_std = torch.clamp(log_std, self.log_std_min, self.log_std_max)
		std = log_std.exp()
		dist = torch.distributions.Normal(mean, std, validate_args=True)
		log_prob = dist.log_prob(torch.atanh(action_batch/self.max_action))
		log_pi = log_prob.sum(dim=1,keepdim=True) - torch.log(1 - (action_batch/self.max_action).pow(2) + 1e-6).sum(dim=1, keepdim=True)
		return log_pi


class Value(nn.Module):
	def __init__(self, state_dim):
		super(Value, self).__init__()

		# Q1 architecture
		self.l1 = nn.Linear(state_dim, 256)
		self.l2 = nn.Linear(256, 256)
		self.l3 = nn.Linear(256, 1)

	def forward(self, state):

		v = F.relu(self.l1(state))
		v = F.relu(self.l2(v))
		v = self.l3(v)
		return v


class AWR(object):
	def __init__(
		self,
		state_dim,
		action_dim,
		max_action,
		discount=0.99,
		tau=0.005,
		policy_noise=0.2,
		noise_clip=0.5,
		policy_freq=2,
		alpha=2.5,
	):

		self.actor = Actor(state_dim, action_dim, max_action).to(device)
		self.actor_target = copy.deepcopy(self.actor)
		self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=3e-4)

		self.value = Value(state_dim).to(device)
		self.value_target = copy.deepcopy(self.value)
		self.value_optimizer = torch.optim.Adam(self.value.parameters(), lr=3e-4)

		self.max_action = max_action
		self.discount = discount
		self.tau = tau
		self.policy_noise = policy_noise
		self.noise_clip = noise_clip
		self.policy_freq = policy_freq
		self.alpha = alpha

		self.total_it   = 0
		self.total_v_it = 0


	def select_action(self, state):
		state = torch.FloatTensor(state.reshape(1, -1)).to(device)
		return self.actor(state,eval=True).cpu().data.numpy().flatten()


	def train(self, replay_buffer, t, batch_size=256, clutch=1e5):

		# Sample replay buffer
		state, action, next_state, reward, not_done, Return = replay_buffer.sample(batch_size)

		v_loss = F.mse_loss(input=self.value(state), target=Return)
		self.value.zero_grad()
		v_loss.backward()
		self.value_optimizer.step()

		# if t > clutch:
		# 	with torch.no_grad():
		# 		v_value = self.value(state)
		# 		rv = Return - v_value
		# 		weight = torch.minimum(torch.exp((rv) / 1.0), torch.ones_like(rv)*20.0)
		#
		# 	mean, log_pi = self.actor(state)
		# 	actor_loss = -(weight * log_pi).mean()
		#
		# 	# Optimize the actor
		# 	self.actor_optimizer.zero_grad()
		# 	actor_loss.backward()
		# 	self.actor_optimizer.step()
		with torch.no_grad():
			v_value = self.value(state)
			rv = Return - v_value
			weight = torch.minimum(torch.exp((rv) / 1.0), torch.ones_like(rv)*20.0)
		log_pi = self.actor.get_log_prob(state, action)

		actor_loss = -torch.mean((weight * log_pi))


		# Optimize the actor
		self.actor_optimizer.zero_grad()
		actor_loss.backward()
		self.actor_optimizer.step()