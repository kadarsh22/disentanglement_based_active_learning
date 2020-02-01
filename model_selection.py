import sys


def model_selection(dataset , gan_type, device,active_learn):

	if dataset == 'mnist':
		sys.path.insert(0, 'utils/mnist/helper_functions_trainer')
		if gan_type == 'dcGAN':

			from dcgan_generate import dcgannmnist

			gan_class = dcgannmnist(device, active_learning=active_learn) ##TODO naming dcgan_mnist
			human_cnn = gan_class.human_cnn()
			active_learner ,optimizer ,scheduler = gan_class.active_learner()

		elif gan_type == 'infoGAN':

			from infogan_generate import infoganmnist

			gan_class = infoganmnist(device, active_learning=active_learn)
			human_cnn = gan_class.human_cnn()
			active_learner ,optimizer ,scheduler = gan_class.active_learner()

		return gan_class, human_cnn, active_learner ,optimizer ,scheduler

	if dataset == 'fashion-mnist':
		sys.path.insert(0, 'utils/fashionmnist/helper_functions_trainer')

		if gan_type == 'dcGAN':

			from dcgan_generate import dcganfashionmnist

			gan_class = dcganfashionmnist(device, active_learning=active_learn)
			human_cnn = gan_class.human_cnn()
			active_learner  ,optimizer ,scheduler = gan_class.active_learner()


		elif gan_type == 'infoGAN':

			from infogan_generate import infoganfashionmnist

			gan_class = infoganfashionmnist(device, active_learning=active_learn)
			human_cnn = gan_class.human_cnn_model()
			active_learner  ,optimizer ,scheduler= gan_class.active_learner()

		return gan_class, human_cnn, active_learner ,optimizer ,scheduler

	if dataset == 'cifar10_2class':
		sys.path.insert(0, 'utils/cifar10_2class/helper_functions_trainer')

		if gan_type == 'dcGAN':
			from dcgan_generate import dcgancifar10_2class
			gan_class = dcgancifar10_2class(device, active_learning=active_learn)
			human_cnn = gan_class.human_cnn()
			active_learner  ,optimizer ,scheduler = gan_class.active_learner()

		elif gan_type == 'infoGAN':
			from infogan_generate import infogancifar10_2class

			gan_class = infogancifar10_2class(device,active_learning=active_learn)
			human_cnn = gan_class.human_cnn_model()
			active_learner  ,optimizer ,scheduler= gan_class.active_learner()
		return gan_class, human_cnn, active_learner ,optimizer ,scheduler