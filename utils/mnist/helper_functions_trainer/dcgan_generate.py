import torch
from torch.autograd import Variable
import torch.nn as nn
import sys
sys.path.insert(0, 'utils/mnist/model_files/dcgan')
sys.path.insert(0, 'utils/mnist/model_files/CNN')
from dcgan import Generator as GAN
from lenet_mnist import Lenet
import torch.nn.functional as F
import numpy as np
import random

class Entropy(nn.Module):
	def __init__(self):
		super(Entropy, self).__init__()

	def forward(self, x):
		b = F.softmax(x, dim=1) * torch.log10(F.softmax(x, dim=1))
		b = b.sum()
		return b

class dcgannmnist:
	def __init__(self, device ,sample_size = 1000 , z_dim = 100,active_learning= False):
		self.device = device
		self.sample_size = sample_size
		self.z_dim = z_dim
		self.active_learning  = active_learning
		self.G = GAN().to(self.device).eval()
		self.G.load_state_dict(torch.load("utils/mnist/trained_models/dcgan/trained_dcgan.pth", map_location=self.device))


	def generate_images(self,model = None):
		z = torch.randn(self.sample_size, self.z_dim, 1, 1)
		z = Variable(z.to(self.device), requires_grad=True).to(self.device)
		optimizer = torch.optim.Adam([z.requires_grad_()], lr= 0.0001)
		if self.active_learning == True:
			z_criterion = Entropy()
			for opt in range(500):
				optimizer.zero_grad()
				samples = F.upsample(self.G(z), 28).to(self.device)
				loss = z_criterion(model(samples))
				loss.backward()
				optimizer.step()
			return samples, loss.item()
		else:
			imgs = F.upsample(self.G(z), 28).to(self.device)
			return imgs

	def human_cnn(self):
		full_cnn_model = Lenet()
		full_cnn_model.load_state_dict(torch.load('utils/mnist/trained_models/humancnn/cnn_lenet_28_tanh.pt', map_location=self.device))
		full_cnn_model.to(self.device)
		return full_cnn_model

	@staticmethod
	def active_learner(device,seed):
		torch.backends.cudnn.deterministic = True
		torch.manual_seed(seed)
		torch.cuda.manual_seed(seed)
		np.random.seed(seed)
		random.seed(seed)
		model = Lenet().to(device)
		optimizer = torch.optim.Adam(model.parameters(), lr= 0.001, amsgrad=True)
		return model , optimizer ,None