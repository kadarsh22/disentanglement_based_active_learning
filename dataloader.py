import torch
import os
import numpy as np
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import sys
sys.path.insert(0, 'utils/')
from Custom_Dataset import NewDataset


# ==================Definition Start======================
def get_loader(batch_size, root, dataset, output_activation):
	data_dir = os.path.join(root, 'data')
	os.makedirs(data_dir, exist_ok=True)
	if output_activation == 'sigmoid':
		transform = transforms.Compose([transforms.ToTensor()])
	else:
		transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize(mean=(0.5,), std=(0.5,))])

	if dataset == 'mnist':
		save_dir = f'{data_dir}/mnist'
		os.makedirs(save_dir, exist_ok=True)
		train_dataset = torchvision.datasets.MNIST(root=f'{data_dir}/mnist', download=True,
												   train=True, transform=transform)

		trainloader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=batch_size,
												  shuffle=True)

		test_dataset = torchvision.datasets.MNIST(root=f'{data_dir}/mnist', download=True,
												  train=False, transform=transform)

		testloader = torch.utils.data.DataLoader(dataset=test_dataset, batch_size=batch_size,
												 shuffle=True)

		dataloader = (trainloader, testloader)
		return dataloader

	elif dataset == 'fashion-mnist':
		save_dir = f'{data_dir}/fashionmnist'
		os.makedirs(save_dir, exist_ok=True)

		train_dataset = torchvision.datasets.FashionMNIST(root=f'{data_dir}/fashionmnist', download=True,
														  train=True, transform=transform)

		trainloader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=batch_size,
												  shuffle=True)

		test_dataset = torchvision.datasets.FashionMNIST(root=f'{data_dir}/fashionmnist', download=True,
														 train=False, transform=transform)

		testloader = torch.utils.data.DataLoader(dataset=test_dataset, batch_size=batch_size,
												 shuffle=True)

		dataloader = (trainloader, testloader)
		return dataloader

	elif dataset == 'cifar10_2class':

		save_dir = f'{data_dir}/cifar10_2classs'
		os.makedirs(save_dir, exist_ok=True)
		transform = transforms.Compose([transforms.ToTensor(),
										transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)), ])
		##TODO train loader should not be done.
		validation_dataset = torchvision.datasets.CIFAR10(root='./data', train=False, transform=transform,
														  download=True)

		validation_dataset.targets = torch.LongTensor(validation_dataset.targets)

		horse_auto_valid_data = [validation_dataset[i][0] for i in range(validation_dataset.data.shape[0]) if
								 validation_dataset.targets[i] == 7 or validation_dataset.targets[i] == 1]
		horse_auto_valid_data = torch.stack(horse_auto_valid_data)
		horse_auto_targets = validation_dataset.targets[
			(np.array(validation_dataset.targets) == 7) | (np.array(validation_dataset.targets) == 1)]

		horse_auto_targets[horse_auto_targets == 7] = 0
		horse_auto_targets[horse_auto_targets == 1] = 1

		validation_dataset = NewDataset(horse_auto_valid_data, horse_auto_targets)

		test_loader = torch.utils.data.DataLoader(dataset=validation_dataset,batch_size=batch_size,
												  shuffle=False, )

		dataloader = (None, test_loader)

		return dataloader
