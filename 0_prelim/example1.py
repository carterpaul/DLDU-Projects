# example1.py

import torch

x = 7*torch.ones(2,3,4)
y = torch.Tensor([i+1 for i in range(24)]).view(2,3,4)

print('x =', x)
print('y =', y)
print('x*y =', x*y)

# exercise: create an 8 by 5 tensor initialized randomly with floats chosen from
# normal distributions with mean, in turn, equal to 1,2...40, and std = 4
means = torch.Tensor([i+1 for i in range(40)])
stds = torch.Tensor([4 for i in range(40)])
tens = torch.normal(means,stds).view(8,5)

print(tens)