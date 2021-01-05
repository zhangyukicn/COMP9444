# spiral.py
# COMP9444, CSE, UNSW

import torch
import torch.nn as nn
import matplotlib.pyplot as plt

class PolarNet(torch.nn.Module):
    def __init__(self, num_hid):
        super(PolarNet, self).__init__()
        self.in_to_hid = nn.Linear(in_features=2,out_features=num_hid,bias=True) #the first layer
        #self.hid_to_out_xxxx = nn.Linear(in_features=num_hid,out_features=num_hid,bias=True) #the second layer
        self.hid_to_out = nn.Linear(in_features=num_hid,out_features=1,bias=True) #the second layer
        self.tan_h1 = None
        # INSERT CODE HERE

    def forward(self, input):
        x= input[:,0]
        y = input[:,1]
        r = torch.sqrt((x**2+y**2)).reshape(-1,1)
        a = torch.atan2(y,x).unsqueeze(1)
        cat_function = torch.cat((r,a),-1)
        hid_sum = self.in_to_hid(cat_function)
        hidden = torch.tanh(hid_sum)
        #hidden = torch.relu(hid_sum)
        self.tan_h1 = hidden
        '''
        xxxxxx = self.hid_to_out_xxxx(hidden)
        hidden1 = torch.tanh(xxxxxx)
        out_sum = self.hid_to_out(hidden1)
        output = torch.sigmoid(out_sum)# CHANGE CODE HERE

        '''
        out_sum = self.hid_to_out(hidden)
        output = torch.sigmoid(out_sum)# CHANGE CODE HERE
        return output

class RawNet(torch.nn.Module):
    def __init__(self, num_hid):
        super(RawNet, self).__init__()
        self.in_to_hid = nn.Linear(in_features=2,out_features=num_hid,bias=True) #the first layer
        self.hid_to_hid = nn.Linear(in_features=num_hid,out_features=num_hid,bias=True) #the second layer
        #self.hid_to_out_xxxx = nn.Linear(in_features=num_hid,out_features=num_hid,bias=True) #the second layer
        self.hid_to_out = nn.Linear(in_features=num_hid,out_features=1,bias=True) #the second layer
        self.tan_h1 = None
        self.tan_h2 = None

    def forward(self, input):
        in_to_hid_x = self.in_to_hid(input)
        hidden1 = torch.tanh(in_to_hid_x)
        #hidden1 = torch.relu(in_to_hid_x)
        self.tan_h1 = hidden1
        hid_to_hid_x = self.hid_to_hid(hidden1)
        hidden2 = torch.tanh(hid_to_hid_x)
        #hidden2 = torch.relu(hid_to_hid_x)
        self.tan_h2 = hidden2
        '''
        id_to_out_xxxx =self.hid_to_out_xxxx(hidden2)
        hidden3 = torch.tanh(id_to_out_xxxx)
        hid_to_out_x = self.hid_to_out(hidden3)
        output = torch.sigmoid(hid_to_out_x)
        '''

        hid_to_out_x = self.hid_to_out(hidden2)
        output = torch.sigmoid(hid_to_out_x)
        return output



def graph_hidden(net, layer, node):
    xrange = torch.arange(start=-7,end=7.1,step=0.01,dtype=torch.float32)
    yrange = torch.arange(start=-6.6,end=6.7,step=0.01,dtype=torch.float32)
    xcoord = xrange.repeat(yrange.size()[0])
    ycoord = torch.repeat_interleave(yrange, xrange.size()[0], dim=0)
    grid = torch.cat((xcoord.unsqueeze(1),ycoord.unsqueeze(1)),1)

    with torch.no_grad(): # suppress updating of gradients
        net.eval()        # toggle batch norm, dropout
        output = net(grid)
        #print(net)
        if layer == 1:
            #print('net',net)
            pred = (net.tan_h1[:,node]>=0).float()
            plt.clf()
            plt.pcolormesh(xrange,yrange,pred.cpu().view(yrange.size()[0],xrange.size()[0]), cmap='Wistia')
        if layer == 2:
            #print('net1',net)
            pred = (net.tan_h2[:,node]>=0).float()
            plt.clf()
            plt.pcolormesh(xrange,yrange,pred.cpu().view(yrange.size()[0],xrange.size()[0]), cmap='Wistia')



