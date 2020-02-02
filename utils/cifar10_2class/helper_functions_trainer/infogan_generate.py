import torch
import numpy as np
import sys
sys.path.insert(0, 'utils/cifar10_2class/model_files/')
sys.path.insert(0, 'utils/cifar10_2class/trained_models/')
from lenet import Lenet
from torch.autograd import  Variable
from infoGAN import Generator
import torch.nn.functional as F
import torch.nn as nn
import random


class Entropy(nn.Module):
    def __init__(self):
        super(Entropy, self).__init__()

    def forward(self, x):
        b = F.softmax(x, dim=1) * torch.log10(F.softmax(x, dim=1))
        b = b.sum()
        return b


class infogancifar10_2class: ##TODO start wih infoganCifar2Helper
	def __init__(self, device, sample_size=1000, z_dim=66, len_discrete_code=2,active_learning =False):
		self.device = device
		self.sample_size = sample_size
		self.z_dim = z_dim
		self.len_discrete_code = len_discrete_code
		self.active_learning = active_learning

		self.G = Generator().to(self.device)
		self.G.load_state_dict(torch.load('utils/cifar10_2class/trained_models/infoGAN/generator_model'))
		self.G.eval()

	def generate_images(self,model = None):
		each_class_samples = int(self.sample_size / self.len_discrete_code)
		z_ = torch.rand((self.sample_size, 62))
		y_cont_ = torch.from_numpy(np.random.uniform(-1, 1, size=(self.sample_size, self.len_discrete_code))).type(
			torch.FloatTensor)
		y_disc_one = torch.Tensor([1, 0] * each_class_samples).view(-1, self.len_discrete_code)
		y_disc_two = torch.Tensor([0, 1] * each_class_samples).view(-1, self.len_discrete_code)
		y_disc_ = torch.cat((y_disc_one, y_disc_two))
		if self.active_learning:
			z_criterion = Entropy()
			optimizer = torch.optim.Adam([z_.requires_grad_()], lr= 0.001)

			for opt in range(500):
				optimizer.zero_grad()
				out_gen = self.G(z_.to(self.device), y_cont_.to(self.device), y_disc_.to(self.device))
				loss = z_criterion(model(out_gen))
				loss.backward()
				optimizer.step()
			return out_gen ,loss.item()
		else:
			imgs = self.G(z_.cuda(), y_cont_.cuda(), y_disc_.cuda()).to(self.device)
			return imgs

	def human_cnn_model(self):
		human_cnn = Lenet()
		human_cnn.load_state_dict(
			torch.load('utils/cifar10_2class/trained_models/humancnn/lenet_weights_0.5_normalisation',
					   map_location=self.device))
		human_cnn.to(self.device)
		return human_cnn

	@staticmethod
	def active_learner(device,seed):
		torch.backends.cudnn.deterministic = True
		torch.manual_seed(seed)
		torch.cuda.manual_seed(seed)
		np.random.seed(seed)
		random.seed(seed)
		model = Lenet().to(device)
		optimizer = torch.optim.Adam(model.parameters(), lr=0.001, amsgrad=True)
		return model, optimizer, None
