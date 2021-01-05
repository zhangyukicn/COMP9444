# kuzu.py
# COMP9444, CSE, UNSW

from __future__ import print_function
import torch
import torch.nn as nn
import torch.nn.functional as F

class NetLin(nn.Module):
    # linear function followed by log_softmax
    def __init__(self):
        super(NetLin, self).__init__()
        self.in_to_out = nn.Linear(in_features=28*28, out_features=10, bias=True)
        # INSERT CODE HERE

    def forward(self, x):
        temp = x.shape[0]
        x = x.view(temp, -1)
        in_x = self.in_to_out(x)
        out_y = F.log_softmax(in_x,dim=1)
        return out_y


class NetFull(nn.Module):
    # two fully connected tanh layers followed by log softmax
    def __init__(self):
        super(NetFull, self).__init__()
        self.in_to_hid = nn.Linear(in_features=28*28,out_features=240,bias=True) #the first layer
        self.hid_to_out = nn.Linear(in_features=240,out_features=10,bias=True) #the second layer
        # INSERT CODE HERE

    def forward(self, x):
        temp = x.shape[0]
        x = x.view(temp, -1)
        hid_sum = self.in_to_hid(x)
        hidden = F.tanh(hid_sum)
        out_sum = self.hid_to_out(hidden)
        output = F.log_softmax(out_sum, dim=None)
        return output


class NetConv(nn.Module):
    # two convolutional layers and one fully connected layer,
    # all using relu, followed by log_softmax
    def __init__(self):
        super(NetConv, self).__init__()
        self.Conv2d1_in_to_hid = nn.Conv2d(in_channels=1,out_channels=32, kernel_size=5) #the first layer
        self.Conv2d1_in_to_Conv2d2 = nn.Conv2d(in_channels=32,out_channels=64, kernel_size=5)#the second layer
        self.pool = nn.MaxPool2d(2)
        self.in_to_hid = nn.Linear(in_features=1024,out_features=240,bias=True) #the fullnet1
        self.hid_to_out = nn.Linear(in_features=240,out_features=10,bias=True) #the fullnet2

    def forward(self, x):
        in_picture_to_hid = self.Conv2d1_in_to_hid(x)
        hid_to_pool1 = self.pool(in_picture_to_hid)
        active_Conv2d1 = F.relu(hid_to_pool1)
        in_picture_to_hid2 = self.Conv2d1_in_to_Conv2d2(active_Conv2d1)
        hid_to_pool2 = self.pool(in_picture_to_hid2)
        active_Conv2d2 = F.relu(hid_to_pool2)
        temp = active_Conv2d2.shape[0]
        active_Conv2d2 = active_Conv2d2.view(temp, -1)
        hid_sum = self.in_to_hid(active_Conv2d2)
        hidden = F.relu(hid_sum)
        out_sum = self.hid_to_out(hidden)
        output = F.log_softmax(out_sum, dim=None)
        return output




