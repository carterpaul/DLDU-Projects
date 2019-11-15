import random
import matplotlib.pyplot as plt
import statistics

sample_size = 20
population_size = 2264
meta_iterations = 1000

results = []

for meta_iteration in range(meta_iterations):
    population = list(range(population_size))

    seen = [0]*population_size
    iteration = 0

    while (sum(seen) < population_size):
        sample = random.sample(population, sample_size)
        for i in sample:
            seen[i] = 1
        iteration += 1

    print("meta_iteration:", meta_iteration)
    results.append(iteration)

print("std: ", statistics.stdev(results))
print("mean:", statistics.mean(results))

plt.hist(results, 50)
plt.show()