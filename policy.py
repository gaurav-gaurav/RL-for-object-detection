import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F 

from torch.distributions import Normal, Categorical
from torchvision import models
from torchsummary import summary

class Attention_module(nn.Module):
	def __init__(self,feature_size=256):
		super(Attention_module, self).__init__()
		self.feature_size = feature_size
		self.attn =  nn.Linear(self.feature_size, self.feature_size, bias = False)

	def forward(self, target):
		return target * F.softmax(self.attn(target), dim=-1)

class CNN(nn.Module):
	def __init__(self):
		super(CNN, self).__init__()
		self.conv1 = nn.Conv2d(3,8,4,2)
		self.relu1 = nn.ReLU(inplace=True)

		self.conv2 = nn.Conv2d(8,16,3,2)
		self.relu2 = nn.ReLU(inplace=True)

		self.conv3 = nn.Conv2d(16,32,3,2)
		self.relu3 = nn.ReLU(inplace=True)

		self.conv4 = nn.Conv2d(32,64,3,2)
		self.relu4 = nn.ReLU(inplace=True)

		self.conv5 = nn.Conv2d(64,128,3,1)
		self.relu5 = nn.ReLU(inplace=True)

		self.conv6 = nn.Conv2d(128,256,3,1)
		self.relu6 = nn.ReLU(inplace=True)

	def forward(self, x):
		x = self.conv1(x)
		x = self.relu1(x)

		x = self.conv2(x)
		x = self.relu2(x)

		x = self.conv3(x)
		x = self.relu3(x)

		x = self.conv4(x)
		x = self.relu4(x)

		x = self.conv5(x)
		x = self.relu5(x)

		x = self.conv6(x)
		x = self.relu6(x)

		return x

class FeatureExtractor(nn.Module):
	def __init__(self,args):
		super(FeatureExtractor, self).__init__()
		if args.pretrain:
			extractor = models.resnet18(pretrained = True)
			self.feature_extractor = nn.Sequential(*(list(extractor.children())[:-1]))
		else:
			self.feature_extractor = CNN()	

	def forward(self,image):
		x = self.feature_extractor(image)
		x = x.view(x.size(0), -1)
		return x

class RL_Module(nn.Module):
	def __init__(self,args,hidden1 = 100, hidden2 = 25, input_size= 2304):
		super(RL_Module, self).__init__()
		self.args = args
		self.input_size = input_size
		self.hidden1 = hidden1
		self.hidden2 = hidden2
		self.action_size = self.args.action_size
		self.action_class = self.args.action_class
		self.action_table = 1/np.linspace(self.args.min_act, self.args.max_act, self.args.action_class)

		if args.action_type == 'continuous':
			self.actor_layer1 = nn.Linear(self.input_size, self.hidden1)
			self.actor_layer2 = nn.Linear(self.hidden1, self.hidden2)
			self.action_mu = nn.Linear(self.hidden2, self.action_size)
			self.action_sig = nn.Parameter(torch.zeros(self.action_size))
			
			self.critic_layer1 = nn.Linear(self.input_size, self.hidden1)
			self.critic_layer2 = nn.Linear(self.hidden1, self.hidden2)
			self.critic = nn.Linear(self.hidden2, 1)

		elif args.action_type == 'discrete':
			self.actor_layer1 = nn.Linear(self.input_size, self.hidden1)
			self.actor_layer2 = nn.Linear(self.hidden1, self.hidden2)
			self.actor = nn.Linear(self.hidden2, self.args.action_class)

			self.critic_layer1 = nn.Linear(self.input_size, self.hidden1)
			self.critic_layer2 = nn.Linear(self.hidden1, self.hidden2)
			self.critic = nn.Linear(self.hidden2, 1)
		
		else:
			raise ValueError('No implemtation for {} action type'.format(self.args.action_type))

	def forward(self, x,y, greed,action=None):
		if self.args.action_type == 'continuous':
			self.action_fwd  = nn.Sequential(self.actor_layer1, nn.ReLU(), self.actor_layer2, nn.ReLU(), self.action_mu)
			self.action_mean = self.action_fwd(x)
			self.action_variance  = torch.exp(self.action_sig.expand_as(self.action_mean))
			self.action_distribution = Normal(loc = self.action_mean, scale = self.action_variance)

			if action==None:
				self.action = self.action_distribution.sample()
			else:
				self.action = action

			self.action_logprob = self.action_distribution.log_prob(self.action)
			self.action_ = torch.clamp(self.action, self.args.min_act, self.args.max_act)

			self.value_fwd = nn.Sequential(self.critic_layer1, nn.ReLU(), self.critic_layer2, nn.ReLU(), self.critic)
			self.value = self.value_fwd(y)

			return self.action, self.action_, self.action_logprob, self.value.squeeze()

		elif self.args.action_type == 'discrete':
			self.action_fwd = nn.Sequential(self.actor_layer1, nn.ReLU(), self.actor_layer2, nn.ReLU(), self.actor)
			self.act_logit = self.action_fwd(x)
			self.action_prob = F.softmax(input = self.act_logit, dim =-1)
			self.action_distribution = Categorical(self.action_prob)

			if action == None:
				if self.args.policy_type == 'greedy':
					epsilon = np.random.uniform(low = 0, high = 1)
					if epsilon >= greed:
						_, self.action = torch.max(self.action_prob,1)
						self.action_ = self.action_table[self.action]
					else:
						_, self.action = torch.max(torch.randint_like(self.action_prob,0, self.args.action_class),1)
						self.action_ = self.action_table[self.action]

				else:
					self.action = self.action_distribution.sample()
					self.action_ = self.action_table[self.action]
			else:
				self.action = action
			self.action_logprob = self.action_distribution.log_prob(self.action)

			self.value_fwd = nn.Sequential(self.critic_layer1, nn.ReLU(), self.critic_layer2, nn.ReLU(), self.critic)
			self.value = self.value_fwd(y)

			return  self.action_, self.action, self.action_logprob, self.value.squeeze()

		else:
			raise ValueError('No implemtation for {} action type'.format(self.args.action_type))

class policy_network(nn.Module):
	def __init__(self, args):
		super(policy_network, self).__init__()
		self.args =  args
		self.feature_actor = FeatureExtractor(self.args)
		self.feature_critic = FeatureExtractor(self.args)
		self.RL_agent = RL_Module(self.args)
		self.attn_layer_actor = Attention_module()
		self.attn_layer_critic = Attention_module()	

	def forward(self, image, greed, act = None):
		x = self.feature_actor(image)
		y = self.feature_critic(image)
		if self.args.attention:
			x = self.attn_layer_actor(x)
			y = self.attn_layer_critic(y)

		action, action_, action_logprob, value = self.RL_agent(x,y,greed,act)
		return action, action_, action_logprob, value
