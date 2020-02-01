import torch
from config import save_config
from config import get_config
from dataloader import get_loader
from trainer import Trainer
from active_learning_dal import ActivelearningDal

def main(config):
	print(config)
	save_config(config)
	data_loader = get_loader(config.batch_size, config.project_root, config.dataset , config.output_activation)
	trainer = Trainer(config, data_loader)
	if config.active_learning == True:
		active_learning = ActivelearningDal(config , data_loader , trainer)
		active_learning.dal_active_learning()
	else:
		trainer.bulk_train()
	return

if __name__ == "__main__":
	config= get_config()
	main(config)