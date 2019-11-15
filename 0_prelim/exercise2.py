import torch
import matplotlib.pyplot as plt

xs = torch.Tensor(40).random_(0,100)
ys = torch.Tensor([2*x+9 for x in xs]) + torch.normal(mean=0.0,std=20.0,size=(1,40))

dm = torch.ones(40,2)
dm[:,1] = xs

w = dm.t().mm(dm).inverse().mm(dm.t()).mm(ys.t()) #y is transposed, contrary to formula
print(w)

plt.scatter(xs.numpy(),ys.numpy())
plt.show()