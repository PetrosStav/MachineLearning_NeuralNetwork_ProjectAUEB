import numpy as np
import math

tests = dict()

# Read all logfiles
for i in [1, 2, 3, 4]:
    for j in range(18):
        with open('./tests_final%d/testrun_%d/testrun_%d.log' % (i, j, j)) as logfile:
            get_next = False
            # Get the line that has the average test costs and create a dictionary that maps
            # the test numbers to the average costs array
            for line in logfile:
                if get_next:
                    tests[(i, j)] = np.asarray([float(e) for e in line[1:-2].split(',')])
                    break
                elif line.startswith('Average test cost'):
                    get_next = True

# Create a copy of the tests dictionary
best = tests.copy()
best_per_test_mnist = dict()
best_per_test_cifar10 = dict()
i = 0
# For every key in the dictionary
for b in best:
    # Get the biggest number and it's index
    best[b] = (np.argmax(best[b]), best[b][np.argmax(best[b])])
    # Initialize everything for each hyperparameter test
    if b[0] != i:
        i += 1
        best_per_test_mnist[i] = (-1, -math.inf)
        best_per_test_cifar10[i] = (-1, -math.inf)
    if b[1] < 9:
        # MNIST
        # Check if bigger than the max and assign accordingly
        if best[b][1] > best_per_test_mnist[i][1]:
            best_per_test_mnist[i] = (b[1], best[b][1])
    else:
        # CIFAR-10
        # Check if bigger than the max and assign accordingly
        if best[b][1] > best_per_test_cifar10[i][1]:
            best_per_test_cifar10[i] = (b[1], best[b][1])

print("MNIST BEST PER TEST -- ", best_per_test_mnist)
print("CIFAR-10 BEST PER TEST -- ", best_per_test_cifar10)
