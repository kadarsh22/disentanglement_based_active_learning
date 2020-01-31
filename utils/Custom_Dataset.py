import torch
from torch.utils.data import Dataset
import pandas

class NewDataset(Dataset):
    def __init__(self, images, labels, transform=None):
        self.images = images
        self.labels = labels
        self.transform = transform


    def __getitem__(self, index):
        img, target = self.images[index].clone(), self.labels[index].clone()

        if self.transform is not None:
            img= self.transform(img)
        return img, target

    def __len__(self):
        return len(self.labels)
    
    def _assign_labels_(self,index, assign_val):
        self.labels[index] = assign_val