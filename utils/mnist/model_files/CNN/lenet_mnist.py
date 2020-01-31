import torch
import torch.nn as nn

class Lenet(nn.Module):
    def __init__(self):
        super(Lenet, self).__init__()
        
        self.cnn1 = nn.Conv2d(in_channels=1, out_channels=32, kernel_size=3, stride=1, padding=1)  
        self.maxpool = nn.MaxPool2d(kernel_size=2)   
        self.cnn2 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, stride=1, padding=1)     
        
        self.relu = nn.ReLU()
        self.tanh = nn.Tanh()
        self.dropout = nn.Dropout(0.8) 
        
        self.fc1 = nn.Linear(64*7*7, 128) 
        self.fc2 = nn.Linear(128, 256)
        self.fc3 = nn.Linear(256, 10)
        

    
    def forward(self, x):
 
        out = self.relu(self.cnn1(x))
        out = self.maxpool(out)
        out = self.relu(self.cnn2(out))
        out = self.maxpool(out)
        out = out.view(out.size(0), -1)
        out = self.tanh(self.fc1(out))
        out = self.dropout(out)
        out = self.tanh(self.fc2(out))
        out = self.dropout(out)
        out = self.fc3(out)
  
        return out