#!/usr/bin/env python3

#Lowest loss achieved: 0.141

import torch
import pandas as pd
import numpy as np
import torch.nn as nn
from DLDUlib import device, train, optimize_ols, center, normalize, r_squared

SAVE_PATH = 'my_implementation.pyt'

# Read the named columns from the csv file into a dataframe.
names = ['SalePrice','1st_Flr_SF','2nd_Flr_SF','Lot_Area','Overall_Qual',
    'Overall_Cond','Year_Built','Year_Remod/Add','BsmtFin_SF_1','Total_Bsmt_SF',
    'Gr_Liv_Area','TotRms_AbvGrd','Bsmt_Unf_SF','Full_Bath']
df = pd.read_csv('AmesHousing.csv', names = names)
data = df.values # read data into a numpy array (as a list of lists)
data = data[1:] # remove the first list which consists of the labels
data = data.astype(np.float) # coerce the entries in the numpy array to floats
data = torch.FloatTensor(data) # convert data to a Torch tensor

data, means = center(data)
data, stdevs = normalize(data)

xss = data[:,1:]
yss = data[:,:1]

class LinearModel(nn.Module):
  
  def __init__(self, means=None, stdevs=None):
    super(LinearModel, self).__init__()
    self.layer1 = nn.Linear(13, 1)

    self.register_buffer('means', means)
    self.register_buffer('stdevs', stdevs)

  def forward(self, xss):
    return self.layer1(xss)

model = LinearModel(means, stdevs).to(device)
criterion = nn.MSELoss()

epochs = 1000
learning_rate = 0.0003549
momentum = 0.899
batchsize = len(data)

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

test_input = torch.FloatTensor([2855, 0, 26690, 8, 7, 1952, 1972, 1040, 2080, 1756, 8, 841, 2])

test_input = (test_input - means[1:]) / stdevs[1:]

test_output = model.forward(test_input).data[0]*stdevs[0]+means[0]

print("Predicted sales price:", test_output.item())

torch.save(model.state_dict(), SAVE_PATH)

# r squared 0.8609
print('1-SSE/SST = {:1.4f}'.format(r_squared(model(xss), yss)))
