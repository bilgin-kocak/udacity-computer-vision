## Define the convolutional neural network architecture

import torch
import torch.nn as nn
import torch.nn.functional as F
# can use the below import should you choose to initialize the weights of your Net
import torch.nn.init as I

class Net(nn.Module):

    def __init__(self):
        super(Net, self).__init__()
        

        ## self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size, stride=1, padding=0)
        # output size = (W-F)/S +1 = (224-5)/1 +1 = 220
        # After convolution: (32, 220, 220)
        # after one pool layer, (32, 110, 110)
        self.conv1 = nn.Conv2d(1, 32, 5)
        
        # maxpool layer
        # pool with kernel_size=2, stride=2
        self.pool = nn.MaxPool2d(2, 2)
        
        #  After convolution: (64,108,108)
        #  after one pool layer, (64, 54, 54)
        self.conv2 = nn.Conv2d(32, 64, 3)
        
        #  After convolution: (128,52,52)
        #  after one pool layer, (128, 26, 26)
        self.conv3 = nn.Conv2d(64, 128, 3)
        
        #  After convolution: (256,24,24)
        #  after one pool layer, (256, 12, 12)
        self.conv4 = nn.Conv2d(128, 256, 3)
        
        #  After convolution: (512,12,12)
        #  after one pool layer, (512, 6, 6)
        self.conv5 = nn.Conv2d(256, 512, 1)
        
        
        # Fully-connected (linear) layers
        self.fc1 = nn.Linear(512*6*6, 1024)
        self.fc2 = nn.Linear(1024, 512)
        self.fc3 = nn.Linear(512, 68*2)
        
        # Dropout  
        self.drop = nn.Dropout(p=0.25)
        
        
    def forward(self, x):
        ## Define the feedforward behavior of this model
        
        # 5 conv/relu + pool layers
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = self.pool(F.relu(self.conv3(x)))
        x = self.pool(F.relu(self.conv4(x)))
        x = self.pool(F.relu(self.conv5(x)))
        
        # Prep for linear layer / Flatten
        x = x.view(x.size(0), -1)
        
        # linear layers with dropout in between
        x = F.relu(self.fc1(x))
        x = self.drop(x)
        x = F.relu(self.fc2(x))
        x = self.drop(x)
        x = self.fc3(x)
        
        return x