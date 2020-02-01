import torch
import torch.nn as nn
from torch.autograd import Variable
import sys
sys.path.insert(0, 'utils/')
import torch.nn.functional as F
from Custom_Dataset import NewDataset
from early_stopping import EarlyStopping
from model_selection import model_selection


class Trainer:
	def __init__(self, config, data_loader):

		self.config = config
		self.input_channels = config.input_channel
		self.input_size = config.input_size
		self.batch_size = config.batch_size
		self.dataset = config.dataset
		self.data_loader = data_loader
		self.device = torch.device("cuda:" + str(config.device_id) if torch.cuda.is_available() else "cpu")

		self.generator, self.human_cnn, self.active_learner, self.optimizer, self.scheduler = model_selection(
			self.dataset, config.gan_type, self.device, config.active_learning)

	def bulk_train(self):

		images = self._generate_images()
		labels = self._human_cnn_annotation(images)
		train_dataset = NewDataset(images, labels)
		_, accuracy = self._train_cnn(train_dataset, self.data_loader[1])

	def _generate_images(self):

		no_of_loops = self.config.data_size / 1000

		for i in range(int(no_of_loops)):
			imgs = self.generator.generate_images(model=None)
			torch.save(imgs, 'image_batch' + str(i))

		data = []
		for i in range(int(no_of_loops)):
			data.append(torch.load('image_batch' + str(i)))

		images = torch.stack(data).view(-1, self.input_channels, self.input_size, self.input_size).to(self.device)

		print('Images Generated Successfully')
		return images

	def _human_cnn_annotation(self, images):

		dummy_labels = torch.LongTensor([8, 7, 0, 4, 5, 3, 6, 2, 9, 1] * int(images.shape[0] / 10)).view(-1).to(
			self.device)

		train_dataset = NewDataset(images, dummy_labels)
		train_loader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=100, shuffle=False)
		self.human_cnn.eval()

		labels = []
		for images, _ in train_loader:
			outputs = F.softmax(self.human_cnn(images.to(self.device)), dim=1)
			_, predicted = torch.max(outputs.data.cpu(), 1)
			labels.append(predicted)

		final_labels = torch.stack(labels).view(-1)

		print('Label annoated by human CNN')

		return final_labels

	def _train_cnn(self, train_dataset, val_dataset):

		train_loader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=self.batch_size, shuffle=True)
		val_loader = torch.utils.data.DataLoader(dataset=val_dataset, batch_size=self.batch_size, shuffle=True)
		test_loader = self.data_loader[2]

		criterion = nn.CrossEntropyLoss()
		model = self.active_learner.to(self.device)

		early_stopping = EarlyStopping(patience=10, verbose=False)
		for epoch in range(50):
			train_loss = []
			valid_loss = []
			model.train()
			for images, labels in train_loader:
				images = Variable(images).to(self.device)
				labels = Variable(labels).to(self.device)
				self.optimizer.zero_grad()
				outputs = model(images)
				loss = criterion(outputs, labels)
				loss.backward()
				self.optimizer.step()
				train_loss.append(loss.item())
			if self.scheduler is not None:
				self.scheduler.step()
			model.eval()

			for images, labels in val_loader:
				images = Variable(images).to(self.device)
				labels = Variable(labels).to(self.device)
				outputs = model(images)
				loss = criterion(outputs, labels)
				valid_loss.append(loss.item())
			train_loss_avg = sum(train_loss) / len(train_loss)
			valid_loss_avg = sum(valid_loss) / len(valid_loss)

			if self.config.active_learning == False:
				print('Epoch: {}, train_loss : {}, test_loss : {}'.format(epoch, train_loss_avg,
																		  valid_loss_avg))
			early_stopping(valid_loss_avg, model)
			if early_stopping.early_stop:
				break

		model.load_state_dict(torch.load('checkpoint.pt'))
		model.eval()

		total = 0.0
		correct = 0.0
		for images, labels in test_loader:
			images = Variable(images).to(self.device)
			labels = Variable(labels).to(self.device)
			outputs = model(images)
			_, predicted = torch.max(outputs.data, 1)
			loss = criterion(outputs, labels)
			total += labels.size(0)
			correct += (predicted.cpu() == labels.cpu()).sum()
			valid_loss.append(loss.item())
		correct = correct.float()
		accuracy = 100 * correct / total


		print('Accuracy on test set {}'.format(accuracy))

		return model, accuracy

#	def get_generation_accuracy(self,images):
