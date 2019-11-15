import torch
import pandas as pd
import numpy as np
import torch.nn as nn
import statistics
from DLDUlib import device, train, optimize_ols, center, normalize, r_squared, cross_validate_train
import copy

names = ['SalePrice','1st_Flr_SF','2nd_Flr_SF','Lot_Area','Overall_Qual',
    'Overall_Cond','Year_Built','Year_Remod/Add','BsmtFin_SF_1','Total_Bsmt_SF',
    'Gr_Liv_Area','TotRms_AbvGrd','Bsmt_Unf_SF','Full_Bath']
df = pd.read_csv('AmesHousing.csv', names = names)
data = df.values # read data into a numpy array (as a list of lists)
data = data[1:] # remove the first list which consists of the labels
data = data.astype(np.float) # coerce the entries in the numpy array to floats
data = torch.FloatTensor(data) # convert data to a Torch tensor

data_train, means = center(data)
data_train, stdevs = normalize(data)

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

folds = 10
bail_after = 10
no_improvement = 0
best_valids = 1e15*torch.ones(folds)

while no_improvement < bail_after:

  model, valids = cross_validate_train(
      k = folds,
      model = model,
      criterion = criterion,
      features = data_train[:,1:],
      targets = data_train[:,:1],
      epochs = epochs,
      learning_rate = learning_rate,
      momentum = momentum,
      batchsize = batchsize,
      verbosity = 1
  )

  print(best_valids.mean().item())
  if valids.mean().item() < best_valids.mean().item():
    best_model = copy.deepcopy(model)
    best_valids = valids
    no_improvement = 0
  else:
    no_improvement += 1
    print("no improvement", no_improvement)

test_input = torch.FloatTensor([2855, 0, 26690, 8, 7, 1952, 1972, 1040, 2080, 1756, 8, 841, 2])

test_input = (test_input - means[1:]) / stdevs[1:]

test_output = best_model.forward(test_input).data[0]*stdevs[0]+means[0]

print("test_output:", test_output)