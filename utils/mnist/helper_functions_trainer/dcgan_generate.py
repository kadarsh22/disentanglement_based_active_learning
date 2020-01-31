import torch
from torch.autograd import Variable
import sys
sys.path.insert(0, 'utils/mnist/model_files/dcgan')
sys.path.insert(0, 'utils/mnist/model_files/CNN')
from dcgan import Generator as GAN
from lenet_mnist import Lenet
import torch.nn.functional as F


class dcgannmnist:
	def __init__(self, device ,sample_size = 1000 , z_dim = 100):
		self.device = device
		self.sample_size = sample_size
		self.z_dim = z_dim
		self.G = GAN().to(self.device).eval()
		self.G.load_state_dict(torch.load("utils/mnist/trained_models/dcgan/trained_dcgan.pth", map_location=self.device))


	def generate_images(self):
		z = torch.randn(self.sample_size, self.z_dim, 1, 1)
		z = Variable(z.to(self.device), requires_grad=True).to(self.device)
		imgs = F.upsample(self.G(z), 28).to(self.device)
		return imgs

	def human_cnn(self):
		full_cnn_model = Lenet()
		full_cnn_model.load_state_dict(torch.load('utils/mnist/trained_models/humancnn/cnn_lenet_28_tanh.pt', map_location=self.device))
		full_cnn_model.to(self.device)
		return full_cnn_model

	def active_learner(self):
		model = Lenet().to(self.device)
		optimizer = torch.optim.Adam(model.parameters(), lr= 0.001, amsgrad=True)
		return model , optimizer ,None