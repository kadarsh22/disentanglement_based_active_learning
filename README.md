# Disentanglement based Active Learning
[Paper](https://ieeexplore.ieee.org/document/9534033)

Disentanglement based Active Learning. \
[Adarsh K](kadarsh22@gmail.com), [Silpa V S](silpavs.43@gmail.com), [S Sumitra](https://www.iist.ac.in/mathematics/sumitra)
*IJCNN 2021*

## Prerequisites
- Ubuntu
- Python 3
- NVIDIA GPU + CUDA CuDNN

<img src='img/dal_block_diagram.jpeg' width=800>

Disentanglement based Active Learning (DAL), a new active learning technique based on self-supervision
which leverages the concept of disentanglement. Instead of requesting labels from human oracle, our method automatically
labels majority of the datapoints, thus drastically reducing the human labeling budget in GAN based active learning approaches


<a name="setup"/>

## Setup

- Clone this repo:
```bash
git clone https://github.com/kadarsh22/disentanglement_based_active_learning.git
cd disentanglement_based_active_learning
```

- Install dependencies:
	- Install dependcies to a new virtual environment.
	```bash
	pip install -r requirements.txt
	```
    
- Download resources:
	- Download pretrained models from [here] (https://drive.google.com/open?id=1M_5ZumHrNjn-_rTiBA6nnsHiYi9p-TbU)
	- Place the pretrained models from here utils/fashionmnist/trainedmoddels 

To run DAL :
```bash
python main.py --dataset  'mnist' --gan_type 'infoGAN' --output_activation 'sigmoid' --data_size 10000
python main.py --dataset  'fashion-mnist' --gan_type 'infoGAN' -output_activation 'tanh' --data_size 10000
```
