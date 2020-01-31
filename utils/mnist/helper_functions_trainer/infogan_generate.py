import torch
from torch.autograd import Variable
import torch.nn as nn
import numpy as np
import sys
sys.path.insert(0, 'utils/mnist/model_files/CNN')
sys.path.insert(0, 'utils/mnist/model_files/infogan')
from lenet_mnist import Lenet
import torch.nn.functional as F
from InfoGAN import InfoGAN

c1_len = 10 # Multinomial
c2_len = 2 # Gaussian
c3_len = 2 # Bernoulli
z_len = 114 # Noise vector length
embedding_len = 128

class Conv2d(nn.Conv2d):
    def reset_parameters(self):
        stdv = np.sqrt(6 / ((self.in_channels  + self.out_channels) * np.prod(self.kernel_size)))
        self.weight.data.uniform_(-stdv, stdv)
        if self.bias is not None:
            self.bias.data.zero_()

class ConvTranspose2d(nn.ConvTranspose2d):
    def reset_parameters(self):
        stdv = np.sqrt(6 / ((self.in_channels  + self.out_channels) * np.prod(self.kernel_size)))
        self.weight.data.uniform_(-stdv, stdv)
        if self.bias is not None:
            self.bias.data.zero_()

class Linear(nn.Linear):
    def reset_parameters(self):
        stdv = np.sqrt(6 / (self.in_features + self.out_features))
        self.weight.data.uniform_(-stdv, stdv)
        if self.bias is not None:
            self.bias.data.zero_()

class Generator(nn.Module):
    def __init__(self):
        super(Generator, self).__init__()
        self.fc1 = Linear(z_len + c1_len + c2_len + c3_len, 1024)
        self.fc2 = Linear(1024, 7 * 7 * 128)

        self.convt1 = ConvTranspose2d(128, 64, kernel_size = 4, stride = 2, padding = 1)
        self.convt2 = ConvTranspose2d(64, 1, kernel_size = 4, stride = 2, padding = 1)

        self.bn1 = nn.BatchNorm1d(1024)
        self.bn2 = nn.BatchNorm1d(7 * 7 * 128)
        self.bn3 = nn.BatchNorm2d(64)

    def forward(self, x):
        x = F.relu(self.bn1(self.fc1(x)))
        x = F.relu(self.bn2(self.fc2(x))).view(-1, 128, 7, 7)

        x = F.relu(self.bn3(self.convt1(x)))
        x = self.convt2(x)

        return F.sigmoid(x)

class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()

        self.conv1 = Conv2d(1, 64, kernel_size = 4, stride = 2, padding = 1) # 28 x 28 -> 14 x 14
        self.conv2 = Conv2d(64, 128, kernel_size = 4, stride = 2, padding = 1) # 14 x 14 -> 7 x 7

        self.fc1 = Linear(128 * 7 ** 2, 1024)
        self.fc2 = Linear(1024, 1)
        self.fc1_q = Linear(1024, embedding_len)

        self.bn1 = nn.BatchNorm2d(128)
        self.bn2 = nn.BatchNorm1d(1024)
        self.bn_q1 = nn.BatchNorm1d(embedding_len)

    def forward(self, x):
        x = F.leaky_relu(self.conv1(x))
        x = F.leaky_relu(self.bn1(self.conv2(x))).view(-1, 7 ** 2 * 128)

        x = F.leaky_relu(self.bn2(self.fc1(x)))
        return self.fc2(x), F.leaky_relu(self.bn_q1(self.fc1_q(x)))



class infoganmnist:
	def __init__(self, device ,sample_size = 1000 , z_dim = 100):
		self.device = device
		self.sample_size = sample_size
		self.z_dim = z_dim
		infogan_gen = Generator().to(device)
		infogan_dis = Discriminator().to(device)
		self.G = InfoGAN(infogan_gen, infogan_dis, embedding_len, z_len, c1_len, c2_len, c3_len, device)
		self.G.load('utils/mnist/trained_models/infogan/infogan_100_z_114/')


	def generate_images(self):
		z_dict = self.G.get_z(c1_len * 100, sequential=True)
		gan_input = torch.cat([z_dict[k] for k in z_dict.keys()], dim=1)
		gan_input = Variable(gan_input, requires_grad=True).to(self.device)
		imgs = self.G.gen(gan_input)
		return imgs

	def human_cnn(self):
		full_cnn_model = Lenet()
		full_cnn_model.load_state_dict(torch.load('utils/mnist/trained_models/humancnn/cnn_lenet_28_sigmoid.pt', map_location=self.device))
		full_cnn_model.eval()
		full_cnn_model.to(self.device)
		return full_cnn_model

	def active_learner(self):
		model = Lenet().to(self.device)
		optimizer = torch.optim.Adam(model.parameters(), lr= 1e-3, amsgrad=True)
		return model , optimizer ,None