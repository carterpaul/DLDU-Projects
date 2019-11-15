from skimage import io
import torch
import pandas as pd
import numpy as np
import torch.nn as nn
from DLDUlib import device, train, optimize_ols, center, normalize, r_squared

digits = io.imread('images/digits.png')  # digits is now a numpy ndarray

xss = torch.Tensor(5000,400)

idx = 0
for i in range(0, 1000, 20):
  for j in range(0, 2000, 20):
    xss[idx] = torch.Tensor((digits[i:i+20,j:j+20]).flatten())
    idx = idx + 1

yss = torch.Tensor(len(xss),1)

for i in range(len(yss)):
  yss[i] = i//500

xss = torch.cat((xss[0:500,:], xss[4000:4500,:]), dim=0)
a = torch.ones((1,), dtype=torch.float32)
b = torch.ones((1,), dtype=torch.float32)
yss = torch.cat((a.new_full((500,), 0), b.new_full((500,), 1))).unsqueeze(1)

class SigmoidModel(nn.Module):

  def __init__(self):
    super(SigmoidModel, self).__init__()
    self.layer1 = nn.Linear(400,1)

  def forward(self, x):
    x = self.layer1(x)
    return torch.sigmoid(x)

model = SigmoidModel()
criterion = nn.MSELoss()

epochs = 1000
learning_rate = 0.001
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
cutoff = 0.5

count = 0
for i in range(len(xss)):
  yhat = model.forward(xss[i]).item()
  y = yss[i].item()
  if (yhat>cutoff and abs(y-eight)<th) or (yhat<cutoff and abs(y-zero)<th):
    count += 1
print("Percentage correct:",100*count/len(xss))