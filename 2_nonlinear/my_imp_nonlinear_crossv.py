import torch
import pandas as pd
import numpy as np
import torch.nn as nn
import statistics
from DLDUlib import device, train, optimize_ols, center, normalize, r_squared

CROSSV_CHUNK = 10

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

losses = []

# iterate through the length of the data by step size CROSSV_CHUNK
for i in range(0, len(data), CROSSV_CHUNK):
  
  if i >= len(data) - CROSSV_CHUNK:
    test_xss = data[i:,1:]
    test_yss = data[i:,:1]
    xss = data[:i,1:]
    yss = data[:i,:1]

  else:
    test_xss = data[i:i+1,1:]
    test_yss = data[i:i+1,:1]
    xss = torch.cat((data[:i,1:], data[i+1:,1:]))
    yss = torch.cat((data[:i:,:1], data[i+1:,:1]))

  class NonLinearModel(nn.Module):

    def __init__(self):
      super(NonLinearModel, self).__init__()
      self.layer1 = nn.Linear(13, 10)
      self.layer2 = nn.Linear(10, 1)

    def forward(self, xss):
      xss = self.layer1(xss)
      xss = torch.relu(xss)
      return self.layer2(xss)

  model = NonLinearModel()
  criterion = nn.MSELoss()

  epochs = 60
  learning_rate = 0.005
  momentum = 0.9
  batchsize = 20

  model = train(
      model,
      criterion,
      features = xss,
      targets = yss,
      epochs = epochs,
      learning_rate = learning_rate,
      momentum = momentum,
      batchsize = batchsize,
      verbosity = 0
  )

  print("\niteration", i // CROSSV_CHUNK)
  print("Test loss: {:1.8f}".format(criterion(model(test_xss), test_yss).item()))
  losses.append(criterion(model(test_xss), test_yss).item())

  if i > 0:
    print("Mean loss:", statistics.mean(losses))
    print("loss std: ", statistics.stdev(losses))