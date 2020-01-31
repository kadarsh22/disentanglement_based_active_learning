import torch
from config import save_config
from config import get_config
from dataloader import get_loader
from trainer import Trainer

def main(config):
	print(config)
	save_config(config)
	data_loader = get_loader(config.batch_size, config.project_root, config.dataset , config.output_activation)
	trainer = Trainer(config, data_loader)
	trainer.bulk_train()
	return

if __name__ == "__main__":
	config= get_config()
	main(config)