import torch
import torch.nn as nn
from torch.autograd import Variable
import sys
import random
sys.path.insert(0, 'utils/fashionmnist/model_files/')
sys.path.insert(0, 'utils/fashionmnist/trained_models/')
from infogan import generator
import human_cnn.fashion as models
import numpy as np
from cnn_classifier import CNNModel
import torch.nn.functional as F


class Entropy(nn.Module):
	def __init__(self):
		super(Entropy, self).__init__()

	def forward(self, x):
		b = F.softmax(x, dim=1) * torch.log10(F.softmax(x, dim=1))
		b = b.sum()
		return b


class infoganfashionmnist:
	def __init__(self, device, sample_size=1000, z_dim=62, len_discrete_code=10, active_learning=False):
		self.device = device
		self.sample_size = sample_size
		self.z_dim = z_dim
		self.active_learning = active_learning
		self.len_discrete_code = len_discrete_code
		self.G = generator().to(self.device)
		self.G.load_state_dict(
			torch.load('utils/fashionmnist/trained_models/infoGAN/infoGAN_G.pkl', map_location=self.device))

	def generate_images(self, model=None):

		z_ = torch.rand((self.sample_size, self.z_dim)).to(self.device)
		z = Variable(z_.to(self.device), requires_grad=True).to(self.device)
		y_ = torch.Tensor([0, 1, 2, 3, 4, 5, 6, 7, 8, 9] * int(self.sample_size / self.len_discrete_code))
		y_disc_ = torch.zeros((self.sample_size, self.len_discrete_code)).scatter_(1,
																				   y_.type(torch.LongTensor).unsqueeze(
																					   1),
																				   1).to(self.device)
		y_cont_ = torch.from_numpy(np.random.uniform(-1, 1, size=(self.sample_size, 2))).type(torch.FloatTensor).to(
			self.device)

		optimizer = torch.optim.Adam([z.requires_grad_()], lr=0.0001)
		if self.active_learning == True:
			z_criterion = Entropy()
			for opt in range(500):
				optimizer.zero_grad()
				samples = self.G(z, y_cont_, y_disc_).to(self.device)
				loss = z_criterion(model(samples))
				loss.backward()
				optimizer.step()
			return samples, loss.item()
		else:
			imgs = self.G(z, y_cont_, y_disc_).to(self.device)
			return imgs

	def human_cnn_model(self):
		human_cnn = models.__dict__['wrn'](num_classes=10, depth=28, widen_factor=10, dropRate=0)
		human_cnn = torch.nn.DataParallel(human_cnn, device_ids=[0])
		checkpoint = torch.load('utils/fashionmnist/trained_models/human_cnn/human_cnn_tanh/model_best.pth.tar',
								map_location=self.device)
		human_cnn.load_state_dict(checkpoint['state_dict'])
		human_cnn.eval()
		human_cnn.to(self.device)
		return human_cnn

	@staticmethod
	def active_learner(device,seed):
		torch.backends.cudnn.deterministic = True
		torch.manual_seed(seed)
		torch.cuda.manual_seed(seed)
		np.random.seed(seed)
		random.seed(seed)
		model = CNNModel().to(device)
		learning_rate = 0.015
		optimizer = torch.optim.Adagrad(model.parameters(), lr=learning_rate)
		scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1)
		return model, optimizer, scheduler
