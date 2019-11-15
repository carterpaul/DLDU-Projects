from skimage import io
import torch
import pandas as pd
import numpy as np
import torch.nn as nn
from DLDUlib import device, train, optimize_ols, center, normalize, r_squared, confusion_matrix

TRAINING_PERCENT = 0.9

digits = io.imread('images/digits.png')  # digits is now a numpy ndarray

xss = torch.Tensor(5000,20,20)

idx = 0
for i in range(0, 1000, 20):
  for j in range(0, 2000, 20):
    xss[idx] = torch.Tensor((digits[i:i+20,j:j+20]))
    idx = idx + 1

yss = torch.LongTensor(len(xss))
for i in range(len(yss)):
  yss[i] = i//500

##### Break of testing and training data #####
indices = torch.randperm(len(xss))
xss = xss.index_select(0, indices)
yss = yss.index_select(0, indices)

cutoff = int(TRAINING_PERCENT * len(xss))
print(cutoff)
xss_train = xss[:cutoff]
yss_train = yss[:cutoff]
xss_test = xss[cutoff:]
yss_test = yss[cutoff:]

# xss_train, means = center(xss_train)
# xss_train, stds = normalize(xss_train)

class ConvolutionalModel(nn.Module):

  def __init__(self):
    super(ConvolutionalModel, self).__init__()
    self.meta_layer1 = nn.Sequential(
        nn.Conv2d(in_channels=1, out_channels=16, kernel_size=5, stride=1, padding = 2),
        nn.ReLU(),
        nn.MaxPool2d(kernel_size = 2, stride = 2, padding = 0)
    )
    self.fc_layer1 = nn.Linear(1600,10)

  def forward(self, xss):
    xss = torch.unsqueeze(xss, dim=1)
    xss = self.meta_layer1(xss)
    xss = torch.reshape(xss, (-1, 1600))
    xss = self.fc_layer1(xss)
    return torch.log_softmax(xss, dim=1)

# create an instance of the model class
model = ConvolutionalModel()

# set the criterion
criterion = nn.NLLLoss()

epochs = 20
learning_rate = 0.001
momentum = 0.9
#batchsize = len(yss_train)//4
batchsize = 30

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

# xss_train = (xss_train - means) + stds
# train_results = torch.argmax(model.forward(xss_train), dim=1)
# count = 0
# for i in range(len(yss_train)):
#   if yss_train[i] == train_results[i]:
#     count += 1
# print("Percentage correct on train set:",100*count/len(xss_train))

# test_results = torch.argmax(model.forward(xss_test), dim=1)
# count = 0
# for i in range(len(yss_test)):
#   if yss_test[i] == test_results[i]:
#     count += 1
# print("Percentage correct on test set: ",100*count/len(xss_test))

pct_correct = confusion_matrix(model(xss_train), yss_train, classes = torch.arange(10), show = True)
print("training", pct_correct)

pct_correct = confusion_matrix(model(xss_test), yss_test, classes = torch.arange(10), show = True)

print("testing", pct_correct)
# Overfitting
# learning rate: 0.01  momentum: 0.9  batchsize: 500 epochs: 500
# Percentage correct on train set: 100.0
# Percentage correct on test set:  86.675