import torch
from torch.utils.data.dataset import random_split
import sys

sys.path.insert(0, 'utils/')
from Custom_Dataset import NewDataset
from model_selection import model_selection
import random
import os
import numpy as np
import matplotlib.pyplot as plt
import torchvision
from copy import deepcopy


class ActivelearningDal:
	def __init__(self, config, data_loader, trainer):

		self.config = config
		self.input_channels = config.input_channel
		self.input_size = config.input_size
		self.batch_size = config.batch_size
		self.dataset = config.dataset
		self.labelling_budget = config.labelling_budget
		self.data_loader = data_loader
		self.no_classes = config.no_classes
		self.active_sample_size = config.active_sample_size
		self.intial_samples = config.initial_samples
		self.device = torch.device("cuda:" + str(config.device_id) if torch.cuda.is_available() else "cpu")
		self.trainer = trainer

		self.generator, self.human_cnn, self.active_learner, self.optimizer, self.scheduler = model_selection(
			self.dataset, config.gan_type, self.device, config.active_learning)

	def dal_active_learning(self):
		random_seeds = [123,22,69,5,108]
		gen_size = int(self.active_sample_size / self.no_classes)  ## reconfirm if its okay to keep it out side loop
		total_active_cycles = int(self.labelling_budget / self.active_sample_size) - 1


		for i in random_seeds:
			print("Executing Random seed " + str(i))
			active_learning_cycle = 0

			self.save_dir = os.path.join(self.config.project_root,
									f'results/{self.config.model_name}/' + 'random_seed' + str(i))
			if not os.path.isdir(self.save_dir):
				self.save_dir = os.path.join(self.config.project_root,
										f'results/{self.config.model_name}/' + 'random_seed' + str(i))
				os.makedirs(self.save_dir, exist_ok=True)

			model = self.active_learner
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

			print(type(train_dataset.data))
			self.save_image(train_dataset.data)
			print(training_data_labels)

			temp_train_dataset = deepcopy(train_dataset)
			num_misclassifications, entropies, properly_classified_data, accuracy_list = ([] for i in range(4))

			size = self.intial_samples
			while (size <= self.labelling_budget):
				model, accuracy = self.trainer._train_cnn(train_dataset, validation_dataset)
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

				if  self.dataset == 'fashion-mnist':
					latent_code = torch.LongTensor([3, 7, 6, 2, 0, 8, 1, 5, 4, 9] * gen_size)
				elif self.dataset == 'mnist':
					latent_code =  torch.LongTensor([0,1,2,3,4,5,6,7,8,9] * gen_size)
				else:
					latent_code = torch.LongTensor([0,1]*gen_size)

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

				self.set_seed(i)
				model = self.active_learner
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

	def save_image(self, image):
		grid_img = torchvision.utils.make_grid(image[:100], nrow=10, normalize=True)
		plt.imshow(grid_img.permute(1, 2, 0).cpu().data)
		plt.savefig(self.save_dir + 'self.config.model_name' + '.png' )

	def set_seed(self, seed):
		torch.backends.cudnn.deterministic = True
		torch.manual_seed(seed)
		torch.cuda.manual_seed(seed)
		np.random.seed(seed)
		random.seed(seed)
