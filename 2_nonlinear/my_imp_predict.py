import torch
import torch.nn as nn

# mean = torch.Tensor([8.1775e+04, 3.7574e+02, 4.2042e+02, 6.3735e+03, 1.4126e+00, 1.0675e+00,
#         3.0768e+01, 2.1162e+01, 4.4367e+02, 4.2064e+02, 4.7722e+02, 1.5279e+00,
#         4.4410e+02, 5.4289e-01])

# stdev = torch.Tensor([1.8383e+05, 1.1595e+03, 3.3222e+02, 9.7601e+03, 6.1714e+00, 5.5764e+00,
#         1.9719e+03, 1.9848e+03, 4.3468e+02, 1.0595e+03, 1.4958e+03, 6.4439e+00,
#         5.7563e+02, 1.5632e+00])

class LinearModel(nn.Module):
  
  def __init__(self, means=torch.Tensor(14), stdevs=torch.Tensor(14)):
    super(LinearModel, self).__init__()
    self.layer1 = nn.Linear(13, 1)

    self.register_buffer('means', means)
    self.register_buffer('stdevs', stdevs)

  def forward(self, xss):
    return self.layer1(xss)

model = LinearModel()

model.load_state_dict(torch.load('my_implementation.pyt'))

test_input = torch.FloatTensor([2855, 0, 26690, 8, 7, 1952, 1972, 1040, 2080, 1756, 8, 841, 2])

test_input = (test_input - model.means[1:]) / model.stdevs[1:]

test_output = model.forward(test_input).data[0]*model.stdevs[0]+model.means[0]

print("Predicted price:", test_output.item())