import torch
import torch.nn as nn
import torchvision
import matplotlib.pyplot as plt

ngpu = 1
batch_size = 100
input_size = 32
nz = 62
nc = 3
ngf = 64
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
class Generator(nn.Module):
    def __init__(self, ngpu,input_size):
        super(Generator, self).__init__()
        self.ngpu = ngpu
        self.input_size = input_size
        
        self.fc = nn.Sequential(
            nn.Linear(nz, 1024),
            nn.BatchNorm1d(1024),
            nn.ReLU(),
            
            nn.Linear(1024, 128 * (input_size // 4) * (input_size // 4)),
            nn.BatchNorm1d(128 * (input_size // 4) * (input_size // 4)),
            nn.ReLU(),
        )
        
        self.main = nn.Sequential(
            # input is Z, going into a convolution
            nn.ConvTranspose2d(ngf * 2,     ngf, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf),
            nn.ReLU(True),
            # state size. (ngf) x 32 x 32
            nn.ConvTranspose2d(    ngf,      nc, 4, 2, 1, bias=False),
            nn.Tanh()
            # state size. (nc) x 64 x 64
        )

    def forward(self, input):
        if input.is_cuda and self.ngpu > 1:
            output = nn.parallel.data_parallel(self.main, input, range(self.ngpu))
        else:
            input = input.view(-1,nz)
            x = self.fc(input)

            x = x.view(-1, 128, (self.input_size // 4), (self.input_size // 4))
            output = self.main(x)
        return output
# netG = Generator(ngpu,input_size).to(device)
# netG.load_state_dict(torch.load('netG_epoch_99.pth'))
#
# noise = torch.randn(batch_size, nz, 1, 1, device=device)
# fake = netG(noise)
# grid_img = torchvision.utils.make_grid(fake.detach().cpu().view(-1,3,32,32), nrow= 10,normalize=True)
# plt.imshow(grid_img.permute(1, 2, 0))
# plt.show()
