import torch
import torch.nn.functional as F
from torch.autograd import Variable
import sys
sys.path.insert(0, 'utils/cifar10_2class/model_files/')
sys.path.insert(0, 'utils/cifar10_2class/trained_models/')
from lenet import Lenet
from generate import Generator


class dcgancifar10_2class:
	def __init__(self, device, sample_size=1000, z_dim=62):
		self.device = device
		self.sample_size = sample_size
		self.z_dim = z_dim
		self.G = Generator(1, input_size=32).to(device)
		self.G.load_state_dict(torch.load('utils/cifar10_2class/trained_models/dcgan/netG_epoch_99.pth', map_location=self.device))

	def generate_images(self):
		z = torch.randn(self.sample_size, self.z_dim, 1, 1)
		z = Variable(z.to(self.device), requires_grad=True).to(self.device)
		imgs = F.upsample(self.G(z), 32)
		return imgs

	def human_cnn(self):
		human_cnn = Lenet()
		human_cnn.load_state_dict(
			torch.load('utils/cifar10_2class/trained_models/humancnn/lenet_weights_0.5_normalisation', map_location=self.device))
		human_cnn.to(self.device)
		return human_cnn

	def active_learner(self):
		model = Lenet().to(self.device)
		optimizer = torch.optim.Adam(model.parameters(), lr = 0.001, amsgrad=True)
		return model, optimizer, None
