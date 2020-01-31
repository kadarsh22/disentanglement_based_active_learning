import torch
from torch.autograd import Variable
import sys
sys.path.insert(0, 'utils/fashionmnist/model_files/')
sys.path.insert(0, 'utils/fashionmnist/trained_models/')
from dcgan import generator as GAN
import human_cnn.fashion as models
from cnn_classifier import CNNModel


class dcganfashionmnist:
	def __init__(self,  device ,sample_size = 1000 , z_dim = 62):
		self.device = device
		self.sample_size = sample_size
		self.z_dim = z_dim
		self.G = GAN(input_dim=62, input_size=28).to(self.device)
		self.G.load_state_dict(torch.load('utils/fashionmnist/trained_models/dcgan/GAN_64_G_tanh_dcgan.pkl', map_location=self.device))


	def generate_images(self):
		z = torch.rand((self.sample_size, self.z_dim)).to(self.device)
		z = Variable(z.to(self.device), requires_grad=True).to(self.device)
		imgs = self.G(z).to(self.device)
		return imgs

	def human_cnn(self):
		human_cnn = models.__dict__['wrn'](num_classes=10, depth=28, widen_factor=10, dropRate=0)
		human_cnn = torch.nn.DataParallel(human_cnn,device_ids=[0])
		checkpoint = torch.load('utils/fashionmnist/trained_models/human_cnn/human_cnn_tanh/model_best.pth.tar',map_location=self.device)
		human_cnn.load_state_dict(checkpoint['state_dict'])
		human_cnn.eval()
		human_cnn.to(self.device)
		return human_cnn

	def active_learner(self):
		model = CNNModel().to(self.device)
		optimizer = torch.optim.Adagrad(model.parameters(), lr=0.015)
		scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1)
		return model ,optimizer ,scheduler

