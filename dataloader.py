import torch
import os
import numpy as np
import torchvision
import torchvision.transforms as transforms
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
		save_dir = f'{data_dir}'
		os.makedirs(save_dir, exist_ok=True)
		train_dataset = torchvision.datasets.MNIST(root=f'{data_dir}/', download=True,
												   train=True, transform=transform)

		train_set, val_set = torch.utils.data.random_split(train_dataset, [50000, 10000])  ## TOTHINK is 50,10 ok?



		test_dataset = torchvision.datasets.MNIST(root=f'{data_dir}/', download=True,
												  train=False, transform=transform)

		testloader = torch.utils.data.DataLoader(dataset=test_dataset, batch_size=batch_size,
												 shuffle=True)

		dataloader = (train_set, val_set, testloader,train_dataset)
		return dataloader

	elif dataset == 'fashion-mnist':
		save_dir = f'{data_dir}'
		os.makedirs(save_dir, exist_ok=True)

		train_dataset = torchvision.datasets.FashionMNIST(root=f'{data_dir}/', download=True,
														  train=True, transform=transform)

		train_set, val_set = torch.utils.data.random_split(train_dataset, [50000, 10000])


		test_dataset = torchvision.datasets.FashionMNIST(root=f'{data_dir}/', download=True,
														 train=False, transform=transform)

		testloader = torch.utils.data.DataLoader(dataset=test_dataset, batch_size=batch_size,
												 shuffle=True)

		dataloader = (train_set, val_set, testloader,train_dataset)
		return dataloader

	elif dataset == 'cifar10_2class':

		save_dir = f'{data_dir}'
		os.makedirs(save_dir, exist_ok=True)
		transform = transforms.Compose([transforms.ToTensor(),
										transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)), ])

		train_dataset = torchvision.datasets.CIFAR10(root=f'{data_dir}/', train=True, transform=transform, download=True)

		dataset_idx = []
		classes_ = [7, 1]
		num_samples_class = int(10000/len(classes_))
		for i in classes_:
			indices = get_same_index(train_dataset.targets, label=torch.tensor(i))
			dataset_idx.append(indices[:num_samples_class])
		image_idx = [item for sublist in dataset_idx for item in sublist]
		train_data = torch.stack([train_dataset[index][0] for index in image_idx])
		labels = torch.LongTensor([[0] * num_samples_class, [1] * num_samples_class]).view(-1)
		train_dataset = NewDataset(train_data, labels)

		validation_dataset = torchvision.datasets.CIFAR10(root=f'{data_dir}/', train=False, transform=transform,
														  download=True)

		validation_dataset.targets = torch.LongTensor(validation_dataset.targets)

		horse_auto_valid_data = [validation_dataset[i][0] for i in range(validation_dataset.data.shape[0]) if
								 validation_dataset.targets[i] == 7 or validation_dataset.targets[i] == 1]
		horse_auto_valid_data = torch.stack(horse_auto_valid_data)
		horse_auto_targets = validation_dataset.targets[
			(np.array(validation_dataset.targets) == 7) | (np.array(validation_dataset.targets) == 1)]

		horse_auto_targets[horse_auto_targets == 7] = 0
		horse_auto_targets[horse_auto_targets == 1] = 1

		val_set = NewDataset(horse_auto_valid_data, horse_auto_targets)


		test_loader = torch.utils.data.DataLoader(dataset=val_set,batch_size=batch_size,
												  shuffle=False, )


		dataloader = (train_dataset, val_set, test_loader,train_dataset)

		return dataloader

def get_same_index(target, label):
    label_indices = []

    for i in range(len(target)):
        if target[i] == label:
            label_indices.append(i)
    return label_indices