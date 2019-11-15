import torch
import pandas as pd
import numpy as np
import torch.nn as nn
from DLDUlib import device, train, optimize_ols, center, normalize, r_squared

TESTING_PERCENT = 0.3

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

# shuffle the data
order = np.array(range(len(data)))
np.random.shuffle(order)
data[np.array(range(len(data)))] = data[order]

# find how many test data points to get
num_tests = int(TESTING_PERCENT*data.size()[0])

# save some data points for testing
test_xss = data[:num_tests,1:]
test_yss = data[:num_tests,:1]

xss = data[num_tests:,1:]
yss = data[num_tests:,:1]

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
)

test_output = model.forward(test_xss)

# r squared 0.9364
print('r squared for training = {:1.4f}'.format(r_squared(model(xss), yss)))

print('r squared for testing  = {:1.4f}'.format(r_squared(test_output, test_yss)))

# un-mean-center and normalize (price is the first variable in the list)
test_output = test_output*stdevs[0]+means[0]
test_yss = test_yss*stdevs[0]+means[0]

# prints out all of the predictions compared to actual
# for i in range(len(test_output)):
#     print("predicted: ${:,.2f}   actual: ${:,.2f}".format(test_output[i].item(), test_yss[i].item()))


# I have to raise the number of nodes in the middle layer to like 100 before I
# start to see much overfitting (in the difference between the r values)

# I use the mean and standard deviation of the whole data set to un-mean-center
# the test data, and to mean center all of the data... is this right?