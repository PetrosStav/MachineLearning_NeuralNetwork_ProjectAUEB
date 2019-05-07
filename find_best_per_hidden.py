import numpy as np
import math

testsM100 = dict()
testsM200 = dict()
testsM300 = dict()

# Read all logfiles
for i in [1, 2, 3, 4]:
    for j in range(18):
        with open('./tests_final%d/testrun_%d/testrun_%d.log' % (i, j, j)) as logfile:
            get_next = False
            # Get the line that has the average test costs and create a dictionary that maps
            # the test numbers to the average costs array
            for line in logfile:
                if get_next:
                    # According to the hidden size M add array to correct dictionary
                    if j % 3 == 0:
                        testsM100[(i, j)] = np.asarray([float(e) for e in line[1:-2].split(',')])
                        break
                    elif j % 3 == 1:
                        testsM200[(i, j)] = np.asarray([float(e) for e in line[1:-2].split(',')])
                        break
                    else:
                        testsM300[(i, j)] = np.asarray([float(e) for e in line[1:-2].split(',')])
                        break
                elif line.startswith('Average test cost'):
                    get_next = True

t = [('M100', testsM100), ('M200', testsM200), ('M300', testsM300)]
for ti in t:
    # Get a copy from the specified dictionary
    best = ti[1].copy()
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

    print("MNIST BEST PER TEST", ti[0], "-- ", best_per_test_mnist)
    print("CIFAR-10 BEST PER TEST", ti[0], "-- ", best_per_test_cifar10)
    print()
