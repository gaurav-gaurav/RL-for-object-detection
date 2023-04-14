import torch
import torch.nn as nn 
from torchvision import models
import torch.nn.functional as F
from torch.distributions import Normal
from torchsummary import summary

class Attention_module(nn.Module):
	def __init__(self,feature_size=512):
		super(Attention_module, self).__init__()
		self.feature_size = feature_size
		self.attn =  nn.Linear(self.feature_size, self.feature_size, bias = False)

	def forward(self, target):
		return target * F.softmax(self.attn(target), dim=-1)

class FeatrueExtractor(nn.Module):
	def __init__(self, args):
		super(FeatrueExtractor, self).__init__()
		self.args = args
		with torch.no_grad():
			self.pretrain = models.resnet18(pretrained = True)

		self.feature_extractor = nn.Sequential(*list(self.pretrain.children())[:-1])

	def forward(self, image):
		x = self.feature_extractor(image)
		state = x.view(x.size(0), -1)
		return state

class ForwardModel(nn.Module):
	def __init__(self,in_size, out_size):
		super(ForwardModel, self).__init__()
		self.in_size = in_size
		self.out_size = out_size
		self.fc1=  nn.Linear(self.in_size, self.out_size)

	def forward(self, x):
		return self.fc1(x)

class InverseModel(nn.Module):
	def __init__(self, in_size, out_size):
		super(InverseModel, self).__init__()
		self.in_size = in_size
		self.out_size = out_size
		self.fc1 = nn.Linear(self.in_size, self.out_size)

	def forward(self,x):
		return self.fc1(x)

class AdversarailHead(nn.Module):
	def __init__(self,feature_size, action_size):
		super(AdversarailHead, self).__init__()
		self.feature_size = feature_size
		self.action_size = action_size

		self.forward_model = ForwardModel(self.action_size + self.feature_size, self.feature_size)
		self.inverse_model = InverseModel(self.feature_size*2, self.action_size)

	def forward(self, a_t, phi_t, phi_t1):
		fwd_in = torch.cat((a_t,phi_t),1)
		inv_in = torch.cat((phi_t, phi_t1),1)

		at_hat = self.inverse_model(inv_in)
		phi_hat_t1 = self.forward_model(fwd_in)
		return at_hat, phi_hat_t1

class ICM_Module(nn.Module):
	def __init__(self, feature_net, action_size, feature_size):
		super(ICM_Module, self).__init__()
		self.feature_net = feature_net
		self.action_size = action_size
		self.feature_size = feature_size

		self.Adversarail_net = AdversarailHead(self.feature_size, self.action_size)

	def forward(self,a_t,s_t,s_t1):
		phi_t = self.feature_net(s_t)
		phi_t1 = self.feature_net(s_t1)

		phi_hat_t1, at_hat = self.Adversarail_net(a_t, phi_t, phi_t1)
		return phi_hat_t1, at_hat, phi_t1, a_t

class A3C_Module(nn.Module):
	def __init__(self, args):
		super(A3C_Module, self).__init__()
		self.action_table = 1/np.linspace(args.min_act,args.max_act,40)
		self.args = args
		self.feature_size = self.args.feature_length
		self.feature_net = FeatrueExtractor()
		self.action_size = self.args.action_size
		self.hidden = 64

		if self.args.attention:
			self.attn_layer = Attention_module()

		if args.action_type == 'continuous':
			self.actor_fc1 = nn.Linear(self.feature_size, self.hidden)
			self.actor_fc2 = nn.Linear(self.hidden, self.action_size)
			self.action_sig = nn.Parameter(torch.zeros(self.action_size))

		elif args.action_type == 'discrete':
			self.actor_fc1 = nn.Linear(self.feature_size, self.hidden)
			self.actor_fc12 = nn.Linear(self.hidden, self.action_class)
		else:
			raise ValueError('No implemtation for {} action type'.format(self.args.action_type))

		self.value_fc1 = nn.Linear(self.feature_size, self.hidden)
		self.value_fc2 = nn.Linear(self.hidden, 1)

	def forward(self,image):
		x = self.feature_net(image)
		if self.args.attention:
			self.x = self.attn_layer(x)

		if self.args.action_type == 'continuous':
			self.action_fwd = nn.Sequential(self.actor_fc1, nn.ReLU(), self.actor_fc2, nn.Sigmoid())
			self.action_mu = 2*self.action_fwd(self.x)
			self.action_variance = torch.exp(self.action_sig.expand_as(self.action_mu))
			self.action_distribution = Normal(loc = sel.action_mu, scale = self.action_variance)
			self.action = self.action_distribution.sample()
			self.action_logprob = self.action_distribution.log_prob(self.action)
			self.action_clamp = torch.clamp(self.action, self.args.min_act, self.args.max_act)

			self.value_fwd = nn.Sequential(self.value_fc1, nn.Tanh(), self.value_fc2, nn.Tanh())
			self.value = self.value_fwd(self.x)

			return self.action_clamp, self.action_logprob, self.value

		elif self.args.action_type == 'discrete':
			self.action_fwd = nn.Sequential(self.actor_fc1, nn.ReLU(), self.actor_fc2, nn.Sigmoid())
			self.act_logit = self.action_fwd(self.x)
			self.action_prob = F.softmax(input = self.act_logit, dim =-1)
			self.action_distribution = Categorical(self.action_prob)
			self.action = self.action_distribution.sample()
			self.action_logprob = self.action_distribution.log_prob(self.action)
			self.action = self.action_table[self.action]

			self.value_fwd = nn.Sequential(self.value_fc1, nn.Tanh(), self.value_fc2, nn.Tanh())
			self.value = self.value_fwd(self.x)

			return self.action, self.value, self.action_logprob

		else:
			raise ValueError('No implemtation for {} action type'.format(self.args.action_type))

	def get_action(self,policy, image):
		image = image.detach().unsqueeze(0)
		action, action_prob, value = policy(image)
		return action, action_prob, value

	def get_a3c_loss(self, reward):
		advantage = (reward.to('self.args.device')- self.value)
		self.policy_loss = (-torch.mean(self.action_logprob,1) * advantage.detach())
		self.value_loss = F.mse_loss(self.value, reward.to(self.args.device))

		self.loss =torch.mean(self.value_loss) + torch.mean(self.policy_loss)
		return self.loss 

	def get_icm_loss(self,image, action, next_image):
		phi_hat_t1,  at_hat, phi_t1, a_t = self.ICM(action, image, next_image)
		self.icm_forward_loss =F.mse_loss(phi_hat_t1, phi_t1)
		self.icm_inverse_loss = F.mse_loss(at_hat, a_t)
		return  self.icm_inverse_loss, self.icm_forward_loss

	def get_loss(self, image, action, next_image, reward):
		inv_loss, fwd_loss = self.get_icm_loss(image, action, next_image)
		a3c_loss = self.get_a3c_loss(reward)
		loss = self.BETA*fwd_loss + (1-self.BETA)*inv_loss + self.LAMBDA*a3c_loss
		return loss
