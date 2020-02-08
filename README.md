# Disentaglement-Active-Learning

To run infogan
```bash
python main.py --dataset  'mnist' --gan_type 'infoGAN' --output_activation 'sigmoid' --data_size 10000
python main.py --dataset  'fashion-mnist' --gan_type 'infoGAN' --output_activation 'tanh' --data_size 10000
python main.py --dataset  'cifar10_2class' --gan_type 'infoGAN' --output_activation 'tanh' --data_size 10000 --no_classes 2 --input_channel 3  --input_size 32
```

Human-CNN model for Fashion-MNIST dataset couldn't be uploaded to Git as size exceeds 100MB. Download  the
model from link below: 
```bash
utils/fashionmnist/trainedmoddels 
link : https://drive.google.com/open?id=1M_5ZumHrNjn-_rTiBA6nnsHiYi9p-TbU
```

Downloaded model should be saved in the folder below:
```bash
utils/fashionmnist/trainedmoddels 
```

Project Directory Structure
```bash
└── utils
    ├── cifar10_2class
    │   ├── helper_functions_trainer
    │   ├── model_files
    │   └── trained_models
    │       ├── dcgan
    │       ├── humancnn
    │       └── infoGAN
    ├── fashionmnist
    │   ├── helper_functions_trainer
    │   ├── model_files
    │   │   └── human_cnn
    │   │       └── fashion
    │   └── trained_models
    │       ├── dcgan
    │       ├── human_cnn
    │       │   ├── human_cnn_sigmoid
    │       │   └── human_cnn_tanh
    │       └── infoGAN
    ├── mnist
    │   ├── code_trained_models
    │   │   ├── CNN
    │   │   ├── DCGAN
    │   │   ├── InfoGAN-100,10,2,2,114
    │   │   ├── VAE_MNIST
    │   │   └── WGAN-GP-mnist
    │   ├── helper_functions_trainer
    │   ├── model_files
    │   │   ├── CNN
    │   │   ├── dcgan
    │   │   └── infogan
    │   └── trained_models
    │       ├── dcgan
    │       ├── humancnn
    │       ├── infogan
    │       │   ├── infogan_100_z_114
    │       │   ├── infogan_mnist-identity_only
    │       │   └── infogan_mnist_identity_style
    │       ├── lenet
    │       └── wgan
    └── svhn
```

LInk to the Arxiv version of paper :
```bash
https://arxiv.org/abs/1912.07018
```
