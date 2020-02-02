import sys

import torch
from torch.utils.data.dataset import random_split

sys.path.insert(0, 'utils/')
from Custom_Dataset import NewDataset
from model_selection import model_selection
from early_stopping import EarlyStopping
import os
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
import torchvision
from torch.autograd import Variable
from copy import deepcopy


class ActivelearningDal:
	def __init__(self, config, data_loader, trainer):

		self.config = config
		self.input_channels = config.input_channel
		self.input_size = config.input_size
		self.active_learning = config.active_learning
		self.batch_size = config.batch_size
		self.dataset = config.dataset
		self.gan_type = config.gan_type
		self.labelling_budget = config.labelling_budget
		self.data_loader = data_loader
		self.num_epochs = config.epoch
		self.no_classes = config.no_classes
		self.active_sample_size = config.active_sample_size
		self.intial_samples = config.initial_samples
		self.device = torch.device("cuda:" + str(config.device_id) if torch.cuda.is_available() else "cpu")
		self.trainer = trainer
		self.generator, self.human_cnn, _, _, _ = model_selection(self.dataset, config.gan_type, self.device,
																  config.active_learning)

	def dal_active_learning(self):
		random_seeds = [123, 22, 69, 5, 108]
		gen_size = int(self.active_sample_size / self.no_classes)  ## reconfirm if its okay to keep it out side loop
		total_active_cycles = int(self.labelling_budget / self.active_sample_size) - 1
		label_freq_cycle = torch.zeros(total_active_cycles, self.no_classes)

		for i in random_seeds:
			print("Executing Random seed " + str(i))
			active_learning_cycle = 0

			self.save_dir = os.path.join(self.config.project_root,
										 f'results/{self.config.model_name}/' + 'random_seed' + str(i))
			if not os.path.isdir(self.save_dir):
				self.save_dir = os.path.join(self.config.project_root,
											 f'results/{self.config.model_name}/' + 'random_seed' + str(i))
				os.makedirs(self.save_dir, exist_ok=True)

			_, _, model, optimizer, scheduler = model_selection(self.dataset, self.gan_type, self.device,
																self.active_learning, i)
			model.to(self.device)

			train_dataset = self.data_loader[3]

			temp_list_data = [train_dataset[i][0] for i in range(len(train_dataset))]
			temp_list_data = torch.stack(temp_list_data)

			temp_list_labels = [train_dataset.targets[i] for i in range(len(train_dataset))]
			temp_list_labels = torch.stack(temp_list_labels)

			train_dataset = NewDataset(temp_list_data, temp_list_labels)
			if self.config.dataset == 'cifar10_2class':
				split_data = random_split(train_dataset, [9000, 1000])
			else:
				split_data = random_split(train_dataset, [50000, 10000])

			temp_train_dataset = deepcopy(split_data[0])
			validation_dataset = deepcopy(split_data[1])

			train_idx = temp_train_dataset.indices
			train_dataset.data = train_dataset.data[train_idx]
			train_dataset.targets = train_dataset.targets[train_idx]
			label_freq_cycle = torch.zeros(total_active_cycles, self.no_classes)
			# Initialisation of training examples

			num_samples_class = int(self.intial_samples / self.no_classes)

			numpy_labels = np.asarray(train_dataset.targets)
			sort_labels = np.sort(numpy_labels)
			sort_index = np.argsort(numpy_labels)
			unique, start_index = np.unique(sort_labels, return_index=True)
			training_index = []
			for s in start_index:
				for i in range(num_samples_class):
					training_index.append(sort_index[s + i])

			train_dataset.data = train_dataset.data[training_index]
			train_dataset.targets = train_dataset.targets[training_index]
			training_data_labels = train_dataset.targets.numpy()

			self.save_image(train_dataset.data)

			temp_train_dataset = deepcopy(train_dataset)
			num_misclassifications, entropies, properly_classified_data, accuracy_list = ([] for i in range(4))
			val_loader = torch.utils.data.DataLoader(dataset=validation_dataset, batch_size=self.batch_size,
													 shuffle=True)

			size = self.intial_samples
			while (size <= self.labelling_budget):
				model, accuracy = self.get_cnn_accuracy(train_dataset, val_loader, model, optimizer, scheduler)
				accuracy_list.append(accuracy)
				print("----Size of training data----", size)
				print("-----Accuracy-------", accuracy)

				label_freq = torch.Tensor(np.unique(training_data_labels, return_counts=True)[1] / size)
				label_freq_cycle[active_learning_cycle] = torch.Tensor(label_freq)

				new_samples, entropy = self.generator.generate_images(model)
				entropies.append(entropy)

				new_samples = new_samples.data.cpu()
				if self.intial_samples == size:
					self.save_image(new_samples)

				if self.dataset == 'fashion-mnist':
					latent_code = torch.LongTensor([3, 7, 6, 2, 0, 8, 1, 5, 4, 9] * gen_size)
				elif self.dataset == 'mnist':
					latent_code = torch.LongTensor([0, 1, 2, 3, 4, 5, 6, 7, 8, 9] * gen_size)
				else:
					latent_code = torch.LongTensor([0, 1] * gen_size)

				if len(properly_classified_data) != 0:
					new_samples = torch.cat((new_samples.data, properly_classified_data.data), 0)
					latent_code = torch.cat((latent_code.cpu(), properly_classified_data.targets), 0)

				data = NewDataset(new_samples, latent_code)

				annotation_loader = torch.utils.data.DataLoader(dataset=data, batch_size=64, shuffle=False,
																drop_last=False)
				predicted = torch.LongTensor().to(self.device)
				for images, labels in annotation_loader:
					images = images.to(self.device)
					outputs = model(images)
					_, predicted_batch = torch.max(outputs.data, 1)
					predicted = torch.cat((predicted, predicted_batch))

				latent_code = latent_code.to(self.device)
				data_diff = latent_code - predicted
				misclassification_index = [i for i, e in enumerate(data_diff) if e != 0]
				properly_classified_index = [i for i, e in enumerate(data_diff) if e == 0]
				len_misclass = len(misclassification_index)
				#				len_prop_index = len(properly_classified_index)
				num_misclassifications.append(len_misclass)

				#         print("Properly classifed index", len_prop_index)
				#         print("MIsclassified index", len_misclass)

				if len_misclass != 0:

					misclassified_data = NewDataset(new_samples[misclassification_index],
													latent_code[misclassification_index])
					humanlabel_loader = torch.utils.data.DataLoader(dataset=misclassified_data,
																	batch_size=64,
																	shuffle=False, drop_last=False)
					annotations = torch.LongTensor().to(self.device)
					for images, labels in humanlabel_loader:
						images = images.to(self.device)
						output = self.human_cnn(images).to(self.device)
						_, annotations_batch = torch.max(output.data, 1)
						annotations = torch.cat((annotations, annotations_batch))

					latent_code[misclassification_index] = annotations

					for index in misclassification_index:
						data._assign_targets_(index, latent_code[index])

				properly_classified_data = NewDataset(new_samples[properly_classified_index].cpu(),
													  latent_code[properly_classified_index].cpu())
				misclassified_data = NewDataset(new_samples[misclassification_index].cpu(),
												latent_code[misclassification_index].cpu())

				temp_train_dataset = torch.utils.data.ConcatDataset((temp_train_dataset, misclassified_data))
				train_dataset = torch.utils.data.ConcatDataset((temp_train_dataset, properly_classified_data))
				training_data_labels = np.append(training_data_labels, np.array(latent_code.cpu()))

				size = len(train_dataset)
				active_learning_cycle = active_learning_cycle + 1
				_, _, model, optimizer, scheduler = model_selection(self.dataset, self.gan_type, self.device,
																	self.active_learning, i)
				model.to(self.device)

				if size % 2000 == 0:
					path = self.save_dir + '/' + 'intermediate_results' + str(size)
					if not os.path.isdir(path):
						os.mkdir(path)
					torch.save(accuracy_list, path + '/accuracy_list')
					torch.save(entropies, path + '/entropies')
					torch.save(label_freq_cycle, path + '/label_frequency')
					torch.save(torch.LongTensor(num_misclassifications), path + '/misclassifications')
					torch.save(train_dataset, path + '/train_dataset')

			print("--------Random seed " + str(i) + "completed--------")

	def get_cnn_accuracy(self, train_dataset, validation_loader, model, optimizer, scheduler):

		train_loader = torch.utils.data.DataLoader(dataset=train_dataset,
												   batch_size=self.batch_size,
												   shuffle=True, drop_last=False)
		criterion = nn.CrossEntropyLoss()
		test_loader = self.data_loader[2]
		early_stopping = EarlyStopping(patience=10, verbose=False)
		for epoch in range(self.num_epochs):
			train_loss = []
			valid_loss = []
			model.train()
			for images, labels in train_loader:
				images = Variable(images).to(self.device)
				labels = Variable(labels).to(self.device)
				optimizer.zero_grad()
				outputs = model(images)
				loss = criterion(outputs, labels)
				loss.backward()
				optimizer.step()
				train_loss.append(loss.item())
			if scheduler is not None:
				scheduler.step()
			model.eval()
			for images, labels in validation_loader:
				images = images.to(self.device)
				labels = labels.to(self.device)
				outputs = model(images)
				loss = criterion(outputs, labels)
				valid_loss.append(loss.item())

			train_loss_avg = sum(train_loss) / len(train_loss)
			valid_loss_avg = sum(valid_loss) / len(valid_loss)

	#		print('Epoch: {}, train_loss : {}, test_loss : {}'.format(epoch, train_loss_avg, valid_loss_avg))
			early_stopping(valid_loss_avg, model)
			if early_stopping.early_stop:
				break

		model.load_state_dict(torch.load('checkpoint.pt'))
		model.eval()

		correct = 0
		total = 0
		test_loss = []
		for images, labels in test_loader:
			images = images.requires_grad_().to(self.device)
			labels = labels.to(self.device)
			outputs = model(images)
			loss = criterion(outputs, labels)
			test_loss.append(loss.data)
			_, predicted = torch.max(outputs.data, 1)
			total += labels.size(0)
			correct += (predicted.cpu() == labels.cpu()).sum()

		correct = correct.float()
		accuracy = 100 * correct / total
		test_loss = sum(test_loss) / len(test_loss)
		print(accuracy)

		return model, accuracy

	def save_image(self, image):
		grid_img = torchvision.utils.make_grid(image[:100], nrow=10, normalize=True)
		plt.imshow(grid_img.permute(1, 2, 0).cpu().data)
		plt.savefig(self.save_dir + 'self.config.model_name' + '.png')
