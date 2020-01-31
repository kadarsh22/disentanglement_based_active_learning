import torch
import numpy as np
import sys
sys.path.insert(0, 'utils/cifar10_2class/model_files/')
sys.path.insert(0, 'utils/cifar10_2class/trained_models/')
from lenet import Lenet
from infoGAN import Generator


class infogancifar10_2class: ##TODO start wih infoganCifar2Helper
	def __init__(self, device, sample_size=1000, z_dim=66, len_discrete_code=2):
		self.device = device
		self.sample_size = sample_size
		self.z_dim = z_dim
		self.len_discrete_code = len_discrete_code

		self.G = Generator().to(self.device)
		self.G.load_state_dict(torch.load('utils/cifar10_2class/trained_models/infoGAN/generator_model'))
		self.G.eval()

	def generate_images(self):
		each_class_samples = int(self.sample_size / self.len_discrete_code)
		z_ = torch.rand((self.sample_size, 62))
		y_cont_ = torch.from_numpy(np.random.uniform(-1, 1, size=(self.sample_size, self.len_discrete_code))).type(
			torch.FloatTensor)
		y_disc_one = torch.Tensor([1, 0] * each_class_samples).view(-1, self.len_discrete_code)
		y_disc_two = torch.Tensor([0, 1] * each_class_samples).view(-1, self.len_discrete_code)
		y_disc_ = torch.cat((y_disc_one, y_disc_two))
		imgs = self.G(z_.cuda(), y_cont_.cuda(), y_disc_.cuda()).to(self.device)
		return imgs

	def human_cnn_model(self):
		human_cnn = Lenet()
		human_cnn.load_state_dict(
			torch.load('utils/cifar10_2class/trained_models/humancnn/lenet_weights_0.5_normalisation',
					   map_location=self.device))
		human_cnn.to(self.device)
		return human_cnn

	def active_learner(self):
		model = Lenet().to(self.device)
		optimizer = torch.optim.Adam(model.parameters(), lr=0.001, amsgrad=True)
		return model, optimizer, None
