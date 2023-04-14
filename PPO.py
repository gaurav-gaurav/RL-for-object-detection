import torch
import torch.nn	as nn 
import torch.nn.functional as F
import numpy as np 

class memory_buffer():
	def __init__(self,args):
		self.args = args
		self.noise_scale_buffer = torch.zeros(args.steps,1)
		self.action_buffer = torch.zeros(args.steps, 1).to(args.device)
		self.action_prob_buffer = torch.zeros(args.steps, 1).to(args.device)
		self.value_buffer = torch.zeros(args.steps, 1).to(args.device)
		self.reward_buffer = torch.zeros(args.steps, 1).to(args.device)

	def update_buffer(self,step, noise_scale, action, action_prob, value):
		self.noise_scale_buffer[step,...] = noise_scale
		self.action_buffer[step,...] = action
		self.action_prob_buffer[step,...] = action_prob
		self.value_buffer[step,...] =  value 

	def fetch_data(self, idx, reward_buffer):
		return self.noise_scale_buffer[idx], self.action_buffer[idx], self.action_prob_buffer[idx], self.value_buffer[idx], reward_buffer[idx]

class ppo():
	def __init__(self, args, clip_value = 0.2):
		self.args = args
		self.clip_value = clip_value

	def get_ppo_loss(self,old_prob, new_prob, mb_gaes, mb_return, new_value):
		ratio = torch.exp(new_prob - old_prob)
		# mb_gaes = (mb_gaes-mb_gaes.mean())/(mb_gaes.std()+ 1e-8)
		pg_loss1 = -mb_return * ratio
		pg_loss2 = -mb_return * torch.clamp(ratio, 1 - self.clip_value, 1 + self.clip_value)
		pg_loss = torch.max(pg_loss1, pg_loss2).mean()

		v_loss = (new_value - mb_return).pow(2).mean()

		loss = pg_loss + v_loss

		return loss

	def get_a2c_loss(self,log_prob, advantage):
		self.policy_loss = (-log_prob* advantage).mean()
		self.value_loss = advantage.pow(2).mean()
		loss = self.policy_loss + self.value_loss
		return loss

	def get_advantage(self,reward, v_pred):
		advantage = [r - v for r, v in zip(reward, v_pred)]
		return torch.FloatTensor(advantage).to(self.args.device)