import argparse
import json
import os
from pathlib import Path


def get_root():
	project_root = str(Path(__file__).resolve().parent.parent)  + '/Disentaglement-Active-Learning-master/'
	return project_root


parser = argparse.ArgumentParser()
parser.add_argument('--gan_type', type=str, default='infoGAN', choices=['dcGAN', 'infoGAN'], help='The type of GAN')
parser.add_argument('--dataset', type=str, default='mnist',
					choices=['mnist', 'fashion-mnist', 'cifar10_2class', 'svhn'],
					help='The name of dataset')
parser.add_argument('--split', type=str, default='', help='The split flag for svhn and stl10')
parser.add_argument('--epoch', type=int, default=1000, help='The number of epochs to run')
parser.add_argument('--batch_size', type=int, default= 64, help='The size of batch')
parser.add_argument('--input_size', type=int, default=28, help='The size of input image')
parser.add_argument('--input_channel', type=int, default=1, help='The size of input image')
parser.add_argument('--save_dir', type=str, default='models', help='Directory name to save the model')
parser.add_argument('--result_dir', type=str, default='results', help='Directory name to save the generated images')
parser.add_argument('--project_root', type=str, default=get_root())
parser.add_argument('--model_name', type=str, default='Debugging')
parser.add_argument('--output_activation', type=str, default='tanh')
parser.add_argument('--device_id', type=int, default=0)
parser.add_argument('--data_size', type=int, default=10000)
parser.add_argument('--z_dim', type=int, default=62)
parser.add_argument('--no_classes', type=int, default=10)

## active learning specfic
parser.add_argument('--initial_samples', type=int, default=100, help='The number of initial datapoints to be labelled')
parser.add_argument('--active_sample_size', type=int, default=200,
					help='The number of points to be labelled in each active learning cycle')
parser.add_argument('--labelling_budget', type=int, default=10000, help='Total number of points that could be labelled')
parser.add_argument('--random_seeds', type=int, default=3, help='Number of random seeds to run for ')
parser.add_argument('--active_learning',type = bool ,default= False ,help = 'Wether active_learning or Not ')
parser.add_argument('--algorithm',type = str ,default= 'gaal' ,help = 'The active learning algorithm which should be ran')


def get_config():
	config = parser.parse_args()
	return config


def save_config(config):
	save_dir = os.path.join(
		config.project_root, f'results/{config.model_name}')
	os.makedirs(save_dir, exist_ok=True)

	with open(f'{save_dir}/config.json', 'w') as fp:
		json.dump(config.__dict__, fp, indent=4, sort_keys=True)

	return
