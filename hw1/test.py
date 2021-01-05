import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import numpy as np
a = torch.arange(4.)
b = torch.reshape(a, (-1, 1))
c = a.unsqueeze(1)
d = a.unsqueeze(-1)
print(b)
print(c)
print(d)



#c = b.reshape((-1,-1))
#print(c)
#input = a[:,1]
#print(input)
