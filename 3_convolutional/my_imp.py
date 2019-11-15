from skimage import io
import torch
import pandas as pd
import numpy as np
import torch.nn as nn
from DLDUlib import device, train, optimize_ols, center, normalize, r_squared

TESTING_PERCENT = 0.2

digits = io.imread('images/digits.png')  # digits is now a numpy ndarray

xss = torch.Tensor(5000,400)

idx = 0
for i in range(0, 1000, 20):
  for j in range(0, 2000, 20):
    xss[idx] = torch.Tensor((digits[i:i+20,j:j+20]).flatten())
    idx = idx + 1

yss = torch.LongTensor(len(xss))
for i in range(len(yss)):
  yss[i] = i//500

##### Break of testing and training data #####
indices = torch.randperm(len(xss))
xss = xss.index_select(0, indices)
yss = yss.index_select(0, indices)

cutoff = int(TESTING_PERCENT * len(xss))
xss_train = xss[:cutoff]
yss_train = yss[:cutoff]
xss_test = xss[cutoff:]
yss_test = yss[cutoff:]

class LogSoftmaxModel(nn.Module):

    def __init__(self):
        super(LogSoftmaxModel, self).__init__()
        self.layer1 = nn.Linear(400,200)
        self.layer2 = nn.Linear(200,200)
        self.layer3 = nn.Linear(200, 10)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.layer1(x)
        x = self.relu(x)
        x = self.layer2(x)
        x = self.relu(x)
        x = self.layer3(x)
        return torch.log_softmax(x, dim=0)

model = LogSoftmaxModel()

criterion = nn.NLLLoss()

epochs = 1000
learning_rate = 0.005
momentum = 0.9
batchsize = len(yss_train)

model = train(
    model,
    criterion,
    features = xss_train,
    targets = yss_train,
    epochs = epochs,
    learning_rate = learning_rate,
    momentum = momentum,
    batchsize = batchsize,
)

count = 0
for i in range(len(xss_test)):
  if  torch.argmax(model.forward(xss_test[i])).item() == yss_test[i].item():
    count += 1
print("Percentage correct on test set:",100*count/len(xss_test))