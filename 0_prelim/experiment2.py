import torch
import matplotlib.pyplot as plt
import statistics

slopes = []
intercepts = []
for i in range(5000):
    xs = torch.Tensor(40).random_(0,100)
    #ys = torch.Tensor([2*x+9 for x in xs]) + torch.normal(mean=0.0,std=20.0,size=(1,40))
    
    #Experiment for question iii: let standard deviation increase from 1 to 40
    ys = torch.Tensor([[2*x+9+torch.normal(mean=0.0,std=x+1,size=(1,1)) for x in xs]])

    dm = torch.ones(40,2)
    dm[:,1] = xs

    w = dm.t().mm(dm).inverse().mm(dm.t()).mm(ys.t()) #y is transposed, contrary to formula
    intercepts.append(w[0].item())
    slopes.append(w[1].item())

slopes_mean = sum(slopes) / float(len(slopes))
slopes_std = statistics.stdev(slopes)
intercepts_mean = sum(intercepts) / float(len(intercepts))
intercepts_std = statistics.stdev(intercepts)

print('distribution      Mean     StDev')
print('       slope {0:>9.4f} {1:>9.4f}'.format(slopes_mean, slopes_std))
print('   intercept {0:>9.4f} {1:>9.4f}'.format(intercepts_mean, intercepts_std))

plt.suptitle('Sampling Distributions')
plt.subplot(1,2,1)
plt.hist(slopes, 50, normed=1)
plt.title("slope")
plt.subplot(1,2,2)
plt.hist(intercepts, 50, normed=1)
plt.title("intercept")
plt.show()

###################################Questions###################################
#   i. They do seem unbiased, averaging aroud 2 and 9, just like the input line
#  ii. Increasing the standard deviation used to generate the noise in the
#      model does not seem to make the estimation biased, but it does increase
#      the standard deviation of the results of the estimation
# iii. Making the standard deviation vary with x (as in line 12) does not
#      affect whether the distributions target the correct value. I think that
#      the standard deviations of the distribution seen in the histogram are
#      just based on the average standard deviation applied when the points are
#      generated. For example, the histograms showing the std varying from 0-40
#      and a fixed std at 20 have similar distributions.