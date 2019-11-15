from skimage import io
import torch
import pandas as pd
import numpy as np
import torch.nn as nn
from DLDUlib import device, train, optimize_ols, center, normalize, r_squared


digits = io.imread('images/digits.png')  # digits is now a numpy ndarray

#x = digits[:20,:20] #  x is now a numpy ndarray of shape 20x20
#x = digits[480:500, 20:40] the 4 that looks like a 9

# for i in range(20):
#     for j in range(20):
#         print('{0:>4}'.format(x[i,j]), end='')
#     print()

xss = torch.Tensor(5000,400)

idx = 0
for i in range(0, 1000, 20):
  for j in range(0, 2000, 20):
    xss[idx] = torch.Tensor((digits[i:i+20,j:j+20]).flatten())
    idx = idx + 1

#yss = torch.LongTensor(len(xss),1)
yss = torch.Tensor(len(xss),1)

for i in range(len(yss)):
  yss[i] = i//500

xss = torch.cat((xss[0:500,:], xss[4000:4500,:]), dim=0)
yss = torch.cat((yss[0:500], yss[4000:4500]), dim=0)

class LinearModel(nn.Module):

  def __init__(self):
    super(LinearModel, self).__init__()
    self.layer1 = nn.Linear(400, 1)

  def forward(self, x):
    return self.layer1(x)

model = LinearModel()
criterion = nn.MSELoss()

epochs = 1000
learning_rate = 0.0000005
momentum = 0.9
batchsize = len(yss)

model = train(
    model,
    criterion,
    features = xss,
    targets = yss,
    epochs = epochs,
    learning_rate = learning_rate,
    momentum = momentum,
    batchsize = batchsize,
)

zero = torch.min(yss).item()
eight = torch.max(yss).item()
th = 1e-3  # threshold
cutoff = (zero+eight)/2

count = 0
for i in range(len(xss)):
  yhat = model.forward(xss[i]).item()
  y = yss[i].item()
  if (yhat>cutoff and abs(y-eight)<th) or (yhat<cutoff and abs(y-zero)<th):
    count += 1
print("Percentage correct:",100*count/len(xss)) #10000 epochs gives you 99.7% accuracy