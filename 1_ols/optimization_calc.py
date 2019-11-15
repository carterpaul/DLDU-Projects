import csv
import torch
import math

#with open('temp_co2_data.csv') as csvfile:
with open('../layered-regression/AmesHousing.csv') as csvfile:
  reader = csv.reader(csvfile, delimiter=',')
  next(csvfile)
  xss, yss  = [], []
  for row in reader:
    xss.append([float(row[2]), float(row[3])])
    yss.append([float(row[1])])

xss = torch.tensor(xss)  # xss is now a 32x2 tensor:  torch.Size([32,2])

xss.sub_(xss.mean(0)); print("centering")
xss.div_(xss.std(0)); print("normalizing")
xss = torch.cat((torch.ones(xss.size()[0],1), xss), 1)  # torch.Size([32,3])

A = xss.t().mm(xss)  # A = X^TX
eigs = [cplx[0].item() for cplx in A.eig()[0]]  # list of the eigenvalues of A
eigs.sort()

print("eigenvalues: " + ', '.join(map(str, eigs)))
print("optimal learning_rate:", 2/(eigs[0]+eigs[-1]))
print("condition number:", eigs[-1]/eigs[0])

print("\n---with momentum---")
print("eigenvalues: " + ', '.join(map(str, eigs)))
print("optimal learning_rate:", (2/(math.sqrt(eigs[0])+math.sqrt(eigs[-1])))**2)
print("optimal momentum:", ((math.sqrt(eigs[-1]) - math.sqrt(eigs[0])) / (math.sqrt(eigs[-1]) + math.sqrt(eigs[0])))**2)
print("condition number:", eigs[-1]/eigs[0])
