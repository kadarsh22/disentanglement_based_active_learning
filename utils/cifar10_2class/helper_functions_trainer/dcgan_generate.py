import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import sys
sys.path.insert(0, 'utils/cifar10_2class/model_files/')
sys.path.insert(0, 'utils/cifar10_2class/trained_models/')
from lenet import Lenet
from generate import Generator



class Entropy(nn.Module):
    def __init__(self):
        super(Entropy, self).__init__()

    def forward(self, x):
        b = F.softmax(x, dim=1) * torch.log10(F.softmax(x, dim=1))
        b = b.sum()
        return b



class dcgancifar10_2class:
	def __init__(self, device, sample_size=1000, z_dim=62,active_learning = False):
		self.device = device
		self.sample_size = sample_size
		self.z_dim = z_dim
		self.active_learning = active_learning
		self.G = Generator(1, input_size=32).to(device)
		self.G.load_state_dict(torch.load('utils/cifar10_2class/trained_models/dcgan/netG_epoch_99.pth', map_location=self.device))

	def generate_images(self,model = None):
		z = torch.randn(self.sample_size, self.z_dim, 1, 1)
		z = Variable(z.to(self.device), requires_grad=True).to(self.device)
		if self.active_learning:
			optimizer = torch.optim.Adam([z.requires_grad_()], lr=0.0001)
			z_criterion = Entropy()
			for opt in range(500):
				optimizer.zero_grad()
				samples = F.upsample(self.G(z), 32).to(self.device)
				loss = z_criterion(model(samples))
				loss.backward()
				optimizer.step()
			return samples, loss.item()
		else:
			imgs = F.upsample(self.G(z), 32)
			return imgs

	def human_cnn(self):
		human_cnn = Lenet()
		human_cnn.load_state_dict(
			torch.load('utils/cifar10_2class/trained_models/humancnn/lenet_weights_0.5_normalisation', map_location=self.device))
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

