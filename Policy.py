import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F 

from torch.distributions import Normal, Categorical
from torchvision import models

class Attention_module(nn.Module):
	def __init__(self,feature_size=512):
		super(Attention_module, self).__init__()
		self.feature_size = feature_size
		self.attn =  nn.Linear(self.feature_size, self.feature_size, bias = False)

	def forward(self, target):
		return target * F.softmax(self.attn(target), dim=-1)

class policy_network(nn.Module):
	def __init__(self, args):
		super(policy_network, self).__init__()

		self.args =  args
		self.layer = self.args.number_of_layers
		self.input_size = args.feature_length
		self.neurons = self.args.number_of_neurons
		self.action_size = self.args.action_size
		self.action_class = self.args.action_class
		self.feature_vector = self.get_CNN()
		self.action_table = 1/np.linspace(args.min_act,args.max_act,40)
		if self.args.attention:
			self.attn_layer = Attention_module()

		if args.action_type == 'continuous':
			self.actor_layer1 = nn.Linear(self.input_size, self.neurons)
			self.actor_layer2 = nn.Linear(self.neurons, self.neurons)
			self.action_mu = nn.Linear(self.neurons, self.action_size)
			self.action_sig = nn.Parameter(torch.zeros(self.action_size))
			
			self.critic_layer1 = nn.Linear(self.input_size, self.neurons)
			self.critic_layer2 = nn.Linear(self.neurons, self.neurons)
			self.critic = nn.Linear(self.neurons,  1)
			

		elif args.action_type == 'discrete':
			self.actor_layer1 = nn.Linear(self.input_size, self.neurons)
			self.actor_layer2 = nn.Linear(self.neurons, self.neurons)
			self.actor = nn.Linear(self.neurons, self.args.action_class)

			self.critic_layer1 = nn.Linear(self.input_size, self.neurons)
			self.critic_layer2 = nn.Linear(self.neurons, self.neurons)
			self.critic = nn.Linear(self.neurons,  1)
		
		else:
			raise ValueError('No implemtation for {} action type'.format(self.args.action_type))


	def forward(self, image):
		x = self.feature_vector(image)
		self.state = x.view(x.size(0), -1)
		if self.args.attention:
			self.state = self.attn_layer(self.state)

		if self.args.action_type == 'continuous':
			self.action_fwd  = nn.Sequential(self.actor_layer1, nn.ReLU(), self.actor_layer2, nn.ReLU(), self.action_mu, nn.Sigmoid())
			self.action_mean = 2*self.action_fwd(self.state)
			self.action_variance  = torch.exp(self.action_sig.expand_as(self.action_mean))
			self.action_distribution = Normal(loc = self.action_mean, scale = self.action_variance)
			self.action = self.action_distribution.sample()
			self.action_logprob = self.action_distribution.log_prob(self.action)
			self.action_clamp = torch.clamp(self.action, self.args.min_act, self.args.max_act)

			self.value_fwd = nn.Sequential(self.critic_layer1, nn.Tanh(), self.critic_layer2, nn.Tanh(), self.critic, nn.Tanh())
			self.value = self.value_fwd(self.state)

			return self.action_clamp, self.action_logprob, self.value

		elif  self.args.action_type == 'discrete':
			self.action_fwd = nn.Sequential(self.actor_layer1, nn.Tanh(), self.actor_layer2, nn.Tanh(), self.actor)
			self.act_logit = self.action_fwd(self.state)
			self.action_prob = F.softmax(input = self.act_logit, dim =-1)
			self.action_distribution = Categorical(self.action_prob)
			self.action = self.action_distribution.sample()
			self.action_logprob = self.action_distribution.log_prob(self.action)
			self.action = self.action_table[self.action]

			self.value_fwd = nn.Sequential(self.critic_layer1, nn.Tanh(), self.critic_layer2, nn.Tanh(), self.critic, nn.Tanh())
			self.value = self.value_fwd(self.state)
			if self.args.eval:
				# print(self.action_prob)
				index = self.action_prob.detach().cpu().numpy().argmax()
				# print(index)
				self.action = self.action_table[index]
			return  self.action, self.action_logprob, self.value

		else:
			raise ValueError('No implemtation for {} action type'.format(self.args.action_type))

	def get_action(self,policy, image):
		image = image.detach().unsqueeze(0)
		action, action_prob, value = policy(image)
		return action, action_prob, value

	def get_pretrain(self, pretrained_model):
		if pretrained_model == 'resnet':	
			return models.resnet18(pretrained = True)
		else:
			raise ValueError('incorrect cnn architecture')

	def get_CNN(self):
		cnn=  self.args.cnn_arch
		if cnn == 'pretrain':
			self.feature_extractor = self.get_pretrain(self.args.pretrain)
			return nn.Sequential(*(list(self.feature_extractor.children())[:-1]))
		else:
			raise ValueError('implemtation only for pretrain networks'.format(self.args.action_type))