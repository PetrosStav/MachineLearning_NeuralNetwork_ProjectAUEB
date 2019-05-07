import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import pickle
import os
import time


# Softplus function
def softplus(a):
    # Calculate softplus using formula that prevents exponent overflows
    return np.log(1 + np.exp(-np.abs(a))) + np.maximum(a, 0)


# Derivative of the softplus function
def softplus_der(a):
    # Calculate the derivative with two different formulas to prevent overflows of exponents
    a1 = np.where(a < 0, 0, a)
    a2 = np.where(a >= 0, 0, a)
    a1 = a1 * (1 / (np.exp(-a1) + 1))
    a2 = a2 * (np.exp(a2) / (1 + np.exp(a2)))
    return a1 + a2


# Tanh function
def tanh(a):
    return np.tanh(a)


# Derivative of the tanh function
def tanh_der(a):
    return 1 - np.power(np.tanh(a), 2)


# Cos function
def cos(a):
    return np.cos(a)


# Derivative of the cos function
def cos_der(a):
    return -np.sin(a)


# Relu function -- EXTRA
def relu(a):
    return np.maximum(a, 0)


# Derivative of the relu function -- EXTRA
def relu_der(a):
    a[a <= 0] = 0
    a[a > 0] = 1
    return a


# Get the derivative of the function h given
def h_der(h, a):
    # Call the right derivative method according to h
    if h == softplus:
        return softplus_der(a)
    elif h == tanh:
        return tanh_der(a)
    elif h == cos:
        return cos_der(a)
    else:
        return relu_der(a)


# Softmax function
# use by default ax=1, when the array is 2D
# use ax=0 when the array is 1D
def softmax(x, ax=1):
    # Find max per row
    m = np.max(x, axis=ax, keepdims=True)
    # Calculate softmax using a formula that subtracts the max per row so that the exponent doesn't overflow
    p = np.exp(x - m)
    soft = p / np.sum(p, axis=ax, keepdims=True)
    return soft


# Function that unpickles a file and returns the dictionary contained
def unpickle(file):
    with open(file, 'rb') as fo:
        dic = pickle.load(fo, encoding='bytes')
    return dic


# Method that loads the mnist dataset
def load_data_mnist():
    """
    Load the MNIST dataset. Reads the training and testing files and create matrices.
    :Expected return:
    train_data:the matrix with the training data
    test_data: the matrix with the data that will be used for testing
    y_train: the matrix consisting of one
                        hot vectors on each row(ground truth for training)
    y_test: the matrix consisting of one
                        hot vectors on each row(ground truth for testing)
    """

    # Load the train files
    df = None
    y_train = []
    for i in range(10):
        tmp = pd.read_csv('data/mnist/train%d.txt' % i, header=None, sep=" ")
        print('Loading train%d.txt' % i)
        # Build labels - one hot vector
        hot_vector = [1 if j == i else 0 for j in range(0, 10)]

        for j in range(tmp.shape[0]):
            y_train.append(hot_vector)
        # Concatenate dataframes by rows
        if i == 0:
            df = tmp
        else:
            df = pd.concat([df, tmp])

    # Get the values from the dataframe
    train_data = df.values
    # Convert the one hot vectors to a numpy array
    y_train = np.array(y_train)

    # Load test files
    df = None
    y_test = []
    for i in range(10):
        tmp = pd.read_csv('data/mnist/test%d.txt' % i, header=None, sep=" ")
        print('Loading test%d.txt' % i)
        # Build labels - one hot vector
        hot_vector = [1 if j == i else 0 for j in range(0, 10)]

        for j in range(tmp.shape[0]):
            y_test.append(hot_vector)
        # Concatenate dataframes by rows
        if i == 0:
            df = tmp
        else:
            df = pd.concat([df, tmp])

    # Get the values from the dataframe
    test_data = df.values
    # Convert the one hot vectors to a numpy array
    y_test = np.array(y_test)

    return train_data, test_data, y_train, y_test


# Method that loads the cifar-10 dataset
def load_data_cifar10():
    """
        Load the CIFAR-10 dataset. Reads the training and testing files and create matrices.
        :Expected return:
        train_data:the matrix with the training data
        test_data: the matrix with the data that will be used for testing
        y_train: the matrix consisting of one
                            hot vectors on each row(ground truth for training)
        y_test: the matrix consisting of one
                            hot vectors on each row(ground truth for testing)
    """

    # Load train files
    train_data = []
    y_train = []
    for i in range(5):
        tmp = unpickle('data/cifar-10-batches-py/data_batch_%d' % (i + 1))
        print('Loading data_batch%d' % (i + 1))
        # Build labels - one hot vector
        for j in range(len(tmp[b'labels'])):
            hot_vector = [1 if n == tmp[b'labels'][j] else 0 for n in range(0, 10)]
            y_train.append(hot_vector)
        # Concatenate by rows
        if i == 0:
            train_data = tmp[b'data']
        else:
            train_data = np.concatenate((train_data, tmp[b'data']), axis=0)
    # Load test file
    tmp = unpickle('data/cifar-10-batches-py/test_batch')
    print('Loading test_batch')
    test_data = tmp[b'data']
    # Build labels - one hot vector
    y_test = []
    for j in range(len(tmp[b'labels'])):
        hot_vector = [1 if n == tmp[b'labels'][j] else 0 for n in range(0, 10)]
        y_test.append(hot_vector)

    # Convert the one hot vectors to a numpy array for train and test
    y_train = np.asarray(y_train)
    y_test = np.asarray(y_test)

    return train_data, test_data, y_train, y_test


# Method that does a forward pass and calculates the cost, gradients and the accuracy of the batch
# Returns the cost of the batch, the gradients for W1 and W2 and the accuracy of the batch
def cost_grads_acc(W1, W2, X, t, h, lamda):
    # Calculate the dot product of X and W1.T
    a = X.dot(W1.T)

    # Use h function given as an activation function for a
    z = h(a)

    # Add 1 as bias to z for 1xM -> 1x(M+1)
    z = np.hstack((np.ones((z.shape[0], 1)), z))

    # Calculate the dot product of z and W2.T
    b = z.dot(W2.T)

    # Calculate the softmax of b
    y = softmax(b)

    # Calculate the cost function from t and y, adding 1e-100 to y, so that the log function doesn't underflow
    Ew = np.sum(t * np.log(y + 1e-100))

    # Calculate the regularization term for the cost function using the norms of W1 and W2 and lamda
    reg = (0.5 * lamda) * (np.sum(np.square(W1)) + np.sum(np.square(W2)))

    # Calculate the regularized cost
    Ew = Ew - reg

    # Calculate the gradients for W2
    gradW2 = (t - y).T.dot(z) - lamda * W2

    # Calculate the derivative of the activation function h for a (X * W1.T)
    d = h_der(h, a)

    # Calculate the gradients for W1
    #  we must remove the first column in W2 in order to have M x (D+1) and not (M+1) x (D+1)
    temp = (t - y).dot(W2[:, 1:]) * d
    gradW1 = temp.T.dot(X) - lamda * W1

    # Get the predictions for the batch
    pred = np.argmax(y, 1)

    # Get the targets for the batch
    targ = np.argmax(t, 1)

    # Calculate accuracy
    acc = np.mean(pred == targ)

    return Ew, gradW1, gradW2, acc


# Method that does a forward pass and calculates the cost and the accuracy of the batch
# Returns the cost of the batch, the predictions for the batch and the accuracy of the batch
def cost_acc(W1, W2, X, t, h, lamda):
    # Calculate the dot product of X and W1.T
    a = X.dot(W1.T)

    # Use h function given as an activation function for a
    z = h(a)

    # Add 1 as bias to z for 1xM -> 1x(M+1)
    z = np.hstack((np.ones((z.shape[0], 1)), z))

    # Calculate the dot product of z and W2.T
    b = z.dot(W2.T)

    # Calculate the softmax of b
    y = softmax(b)

    # Calculate the cost function from t and y, adding 1e-100 to y, so that the log function doesn't underflow
    Ew = np.sum(t * np.log(y + 1e-100))

    # Calculate the regularization term for the cost function using the norms of W1 and W2 and lamda
    reg = (0.5 * lamda) * (np.sum(np.square(W1)) + np.sum(np.square(W2)))

    # Calculate the regularized cost
    Ew = Ew - reg

    # Get the predictions for the batch
    pred = np.argmax(y, 1)

    # Get the targets for the batch
    targ = np.argmax(t, 1)

    # Calculate accuracy
    acc = np.mean(pred == targ)

    return Ew, pred, acc


# Method tha calculates the cost of a batch -- only used in grad_check
# Return the cost of the batch
def calc_Ew(W1, W2, X, t, h, lamda):
    # Calculate the dot product of X and W1.T
    a = X.dot(W1.T)

    # Use h function given as an activation function for a
    z = h(a)

    # Add 1 as bias to z so that 1xM -> 1x(M+1)
    z = np.hstack((np.ones((z.shape[0], 1)), z))

    # Calculate the dot product of z and W2.T
    b = z.dot(W2.T)

    # Calculate the softmax of b
    y = softmax(b)

    # Calculate the cost function from t and y, adding 1e-100 to y, so that the log function doesn't underflow
    Ew = np.sum(t * np.log(y + 1e-100))

    # Calculate the regularization term for the cost function using the norms of W1 and W2 and lamda
    reg = (0.5 * lamda) * (np.sum(np.square(W1)) + np.sum(np.square(W2)))

    # Calculate the regularized cost
    Ew = Ew - reg

    return Ew


# Method that performs grad checking, in order to evaluate the correctness of the gradient calculation in the model
def grad_check(W1, W2, X, t, h, lamda):
    # Define epsilon
    epsilon = 1e-6

    # Get a sample of 5 instances
    _list = np.random.randint(X.shape[0], size=5)
    x_sample = np.array(X[_list, :])
    t_sample = np.array(t[_list, :])

    # Calculate gradW1 and gradW2 using the mathematical formulas of backpropagation and the chain rule
    _, gradW1, gradW2, _ = cost_grads_acc(W1, W2, x_sample, t_sample, h, lamda)

    # Print the shapes of the two matrices
    print("gradW1 shape:", gradW1.shape)
    print("gradW2 shape:", gradW2.shape)

    # Initialize the numericalGradW1 and numericalGradW1 to zeroes using the shapes of gradW1 and gradW2
    numericalGradW1 = np.zeros(gradW1.shape)
    numericalGradW2 = np.zeros(gradW2.shape)

    # Compute all numerical gradient estimates and store them in
    # the matrices numericalGradW1 & numericalGradW2

    # For numericalGradW1
    print('numericalGradW1...')
    for k in range(numericalGradW1.shape[0]):
        # Print K at every 5 iterations
        if k % 5 == 0:
            print('K: %d' % k)
        for d in range(numericalGradW1.shape[1]):
            # Add epsilon to the w[k,d]
            w_tmp = np.copy(W1)
            w_tmp[k, d] += epsilon
            # Calculate cost using the forward pass
            e_plus = calc_Ew(w_tmp, W2, x_sample, t_sample, h, lamda)

            # Subtract epsilon to the w[k,d]
            w_tmp = np.copy(W1)
            w_tmp[k, d] -= epsilon
            # Calculate cost using the forward pass
            e_minus = calc_Ew(w_tmp, W2, x_sample, t_sample, h, lamda)

            # Approximate gradient ( E[ w[k,d] + theta ] - E[ w[k,d] - theta ] ) / 2*e
            numericalGradW1[k, d] = (e_plus - e_minus) / (2 * epsilon)

    # For numericalGradW2
    print('numericalGradW2...')
    for k in range(numericalGradW2.shape[0]):
        # Print K at every 5 iterations
        if k % 5 == 0:
            print('K: %d' % k)
        for d in range(numericalGradW2.shape[1]):
            # Add epsilon to the w[k,d]
            w_tmp = np.copy(W2)
            w_tmp[k, d] += epsilon
            # Calculate cost using the forward pass
            e_plus = calc_Ew(W1, w_tmp, x_sample, t_sample, h, lamda)

            # Subtract epsilon to the w[k,d]
            w_tmp = np.copy(W2)
            w_tmp[k, d] -= epsilon
            # Calculate cost using the forward pass
            e_minus = calc_Ew(W1, w_tmp, x_sample, t_sample, h, lamda)

            # Approximate gradient ( E[ w[k,d] + theta ] - E[ w[k,d] - theta ] ) / 2*e
            numericalGradW2[k, d] = (e_plus - e_minus) / (2 * epsilon)

    return gradW1, gradW2, numericalGradW1, numericalGradW2


# Method that loads the dataset, preprocess it and runs the model using the given parameters and hyperparameters
def run_model(dataset_choice=0, shuffle_data_first=True, hidden_size=100, batch_size=100, learning_rate=0.001,
              lam_reg=0.1, activation=tanh, per_batch_print=50, weight_init_mode='normal', grad_check_enabled=False,
              plot_sample_instances=False, epochs=200, per_epoch_shuffle_interval=0, plot_false_predicted_interval=0,
              plot_learning_graphs=True, save_graph_plot=False, dataset=None, verbose=False, converge_thres=None):
    """ Load Datasets """

    cifar10_labels = None

    if dataset is None:
        # 0 for mnist
        # 1 for cifar-10
        # load_data_cifar10() or load_data_mnist()
        if dataset_choice == 0:
            X_train, X_test, y_train, y_test = load_data_mnist()
        else:
            X_train, X_test, y_train, y_test = load_data_cifar10()
            tmp = unpickle('data/cifar-10-batches-py/batches.meta')
            cifar10_labels = [e.decode('utf-8') for e in tmp[b'label_names']]
    else:
        print('Using assigned dataset...')
        if len(dataset) == 4:
            X_train, X_test, y_train, y_test = dataset
        else:
            X_train, X_test, y_train, y_test, cifar10_labels = dataset
            dataset_choice = 1

    """ Preprocess data """

    # Project data from [0, 255] to [0, 1]
    X_train = X_train.astype(float) / 255
    X_test = X_test.astype(float) / 255

    # Stack 1 as bias at input
    # ex. shape from (60000,784)   -> (60000,785)
    #                (50000, 3072) -> (50000, 3073)
    X_train = np.hstack((np.ones((X_train.shape[0], 1)), X_train))
    X_test = np.hstack((np.ones((X_test.shape[0], 1)), X_test))

    """ Shuffle datasets """

    if shuffle_data_first:

        # Shuffle train data
        test = [(x, y) for x, y in zip(X_train, y_train)]
        np.random.shuffle(test)

        X_train = [x for (x, y) in test]
        y_train = [y for (x, y) in test]

        X_train = np.asarray(X_train)
        y_train = np.asarray(y_train)

        # Shuffle test data
        test = [(x, y) for x, y in zip(X_test, y_test)]
        np.random.shuffle(test)

        X_test = [x for (x, y) in test]
        y_test = [y for (x, y) in test]

        X_test = np.asarray(X_test)
        y_test = np.asarray(y_test)

    """ Initialize weights / parameters """

    # Initialize weights according to the mode the user chose
    if weight_init_mode == 'uniform':
        # Uniform initialization
        W1 = np.random.uniform(0, 1, (hidden_size, X_train.shape[1]))
        W2 = np.random.uniform(0, 1, (y_train.shape[1], hidden_size + 1))
    elif weight_init_mode == 'normal':
        # Normal initialization with center and radius according to inputs
        center = 0
        r = 1
        W1 = np.random.normal(center, r, (hidden_size, X_train.shape[1])) * np.sqrt(1 / X_train.shape[1])
        W2 = np.random.normal(center, r, (y_train.shape[1], hidden_size + 1)) * np.sqrt(1 / (hidden_size + 1))
    elif weight_init_mode == 'xavier':
        # Xavier Glorot normal initialization ( inputs and outputs ) -- for tanh
        W1 = np.random.normal(0, 1, (hidden_size, X_train.shape[1])) * np.sqrt(1 / (X_train.shape[1] + hidden_size))
        W2 = np.random.normal(0, 1, (y_train.shape[1], hidden_size + 1)) * np.sqrt(
            1 / (hidden_size + 1 + y_train.shape[1]))
    elif weight_init_mode == 'normal_relu':
        # Normal initialization with center and radius according to inputs -- for relu -- EXTRA
        center = 0
        r = 1
        W1 = np.random.normal(center, r, (hidden_size, X_train.shape[1])) * np.sqrt(1 / (2 * X_train.shape[1]))
        W2 = np.random.normal(center, r, (y_train.shape[1], hidden_size + 1)) * np.sqrt(1 / (2 * hidden_size + 1))

    """ Perform Grad Checking """

    if grad_check_enabled:
        # Get the mathematical and the numerical grads for W1 and W2
        gradW1, gradW2, numericalGradW1, numericalGradW2 = grad_check(W1, W2, X_train, y_train, activation, lam_reg)
        print("Grad Check difference for gradW1")
        print(np.abs(gradW1 - numericalGradW1))
        print("Difference of gradW1 and numericalGradW1 using Euclidean distances (Andrew Ng):")
        print(np.linalg.norm(gradW1 - numericalGradW1) / (np.linalg.norm(gradW1) + np.linalg.norm(numericalGradW1)))
        print("Grad Check difference for gradW2")
        print(np.abs(gradW2 - numericalGradW2))
        print("Difference of gradW2 and numericalGradW2 using Euclidean distances (Andrew Ng):")
        print(np.linalg.norm(gradW2 - numericalGradW2) / (np.linalg.norm(gradW2) + np.linalg.norm(numericalGradW2)))

    """ Plot a sample of the instances"""

    if plot_sample_instances:
        if dataset_choice == 0:
            # Plot 100 random images from the training set MNIST
            # Create a 10 by 10 plot
            n = 100
            sqrt_n = int(n ** 0.5)
            samples = np.random.randint(X_train.shape[0], size=n)

            plt.figure(figsize=(11, 11))

            cnt = 0
            for i in samples:
                cnt += 1
                plt.subplot(sqrt_n, sqrt_n, cnt)
                plt.subplot(sqrt_n, sqrt_n, cnt).axis('off')
                # Reshape instance to a 28 by 28 by 1 (Grayscale) image and plot it
                plt.imshow(X_train[i][1:].reshape(28, 28), cmap='gray')
                plt.title(str(np.argmax(y_train, 1)[i]))

            plt.show()
        else:
            # Plot 100 random images from the training set CIFAR-10
            # Create a 10 by 10 plot
            n = 100
            sqrt_n = int(n ** 0.5)
            samples = np.random.randint(X_train.shape[0], size=n)

            plt.figure(figsize=(11, 11))

            cnt = 0
            for i in samples:
                cnt += 1
                plt.subplot(sqrt_n, sqrt_n, cnt)
                plt.subplot(sqrt_n, sqrt_n, cnt).axis('off')
                # Reshape instance to a 32 by 32 by 3 (RGB) image and plot it
                X_train_r = X_train[i][1:][:1024].reshape(32, 32)
                X_train_g = X_train[i][1:][1024:2048].reshape(32, 32)
                X_train_b = X_train[i][1:][2048:].reshape(32, 32)
                X_train_ = np.stack((X_train_r, X_train_g, X_train_b), axis=2)
                plt.imshow(X_train_)
                plt.title(str(cifar10_labels[np.argmax(y_train, 1)[i]]))

            plt.show()

    """ Run model for the epochs specified by the user """

    costs_train = []
    costs_test = []
    accus_train = []
    accus_test = []

    print('Training the model:\n')
    print('Dataset: %s -- Hidden Size: %d -- Batch Size: %d' % (
        ('MNIST' if dataset_choice == 0 else 'CIFAR-10'), hidden_size, batch_size))
    print('Activation: %s -- Learning Rate: %g -- Lambda: %g' % (activation.__name__, learning_rate, lam_reg))
    print('Epochs: %d\n' % epochs)

    # For convergence checks
    last_cost = None
    converged = False

    for epoch in range(1, epochs + 1):
        # If the model has converged, stop the training
        if converge_thres is not None and converged:
            print('\n@@ Model converged at epoch %d @@' % (epoch-1))
            break

        if verbose:
            print("\n---Epoch %d---\n" % epoch)
        else:
            print('\rProgress %.2f%%' % ((epoch/epochs)*100), end='')
            if epoch == epochs:
                print()

        """ Train model """

        if verbose:
            print("Training...")

        """ Shuffle train set every interval given """

        if per_epoch_shuffle_interval != 0 and epoch % per_epoch_shuffle_interval == 0:
            if verbose:
                print("Shuffling train dataset")

            test = [(x, y) for x, y in zip(X_train, y_train)]
            np.random.shuffle(test)

            X_train = [x for (x, y) in test]
            y_train = [y for (x, y) in test]

            X_train = np.asarray(X_train)
            y_train = np.asarray(y_train)

        """ Train model for every batch -- forward pass, calculate gradients, update weights"""

        # Start of train epoch
        costs = []
        accus = []
        for batch in range(0, X_train.shape[0], batch_size):
            # Get the batch from the train data
            X_train_b = X_train[batch:batch + batch_size]
            y_train_b = y_train[batch:batch + batch_size]
            # Calculate the cost, grads and accuracy of the batch
            Ew, gradW1, gradW2, acc = cost_grads_acc(W1, W2, X_train_b, y_train_b, activation, lam_reg)
            # Add the batch cost and accuracy to the epoch's list
            costs.append(Ew)
            accus.append(acc)

            if verbose and per_batch_print != 0 and batch % (per_batch_print * batch_size) == 0:
                print(
                    "Batch %d :--: Cost function: %f :--: Accuracy: %f :--: Average cost: %f :--: Average accuracy: %f" % (
                        batch / batch_size, Ew, acc, np.mean(costs), np.mean(accus)))
            # Update the weights using stochastic gradient ascend using the learning rate as a factor
            W1 += learning_rate * gradW1
            W2 += learning_rate * gradW2
        # Add the average cost and accuracy of the epoch to the lists
        costs_train.append(np.mean(costs))
        accus_train.append(np.mean(accus))

        # End of train epoch
        if verbose:
            print(
                "\n@@ Train Epoch %d :--: Average cost: %f :--: Average accuracy: %f @@" % (
                    epoch, np.mean(costs), np.mean(accus)))

        """ Test model for every batch -- forward pass"""

        # Start of test epoch
        if verbose:
            print("\nTesting...")
        costs = []
        accus = []
        pred = []
        for batch in range(0, X_test.shape[0], batch_size):
            # Get the batch from the test data
            X_test_b = X_test[batch:batch + batch_size]
            y_test_b = y_test[batch:batch + batch_size]
            # Calculate the cost and accuracy of the batch
            Ew, pred_, acc = cost_acc(W1, W2, X_test_b, y_test_b, activation, lam_reg)
            # Add the batch cost, predictions and accuracy to the epoch's list
            costs.append(Ew)
            accus.append(acc)
            pred.append(pred_)

            if verbose and per_batch_print != 0 and batch % (per_batch_print * batch_size) == 0:
                print(
                    "Batch %d :--: Cost function: %f :--: Accuracy: %f :--: Average cost: %f :--: Average accuracy: %f" % (
                        batch / batch_size, Ew, acc, np.mean(costs), np.mean(accus)))

        # Find if the epoch test cost has converged
        epoch_cost = np.mean(costs)
        if converge_thres is not None:
            if last_cost is None:
                last_cost = epoch_cost
            else:
                if abs(epoch_cost - last_cost) <= converge_thres:
                    converged = True
                else:
                    last_cost = epoch_cost
        # Add the average cost and accuracy of the epoch to the lists
        costs_test.append(epoch_cost)
        accus_test.append(np.mean(accus))

        # End of test epoch
        if verbose:
            print(
                "\n@@ Test Epoch %d :--: Average cost: %f :--: Average accuracy: %f @@" % (
                    epoch, np.mean(costs), np.mean(accus)))

        """ Plot the instances that were false predicted every interval given """

        if plot_false_predicted_interval != 0 and epoch % plot_false_predicted_interval == 0:
            # Find all the false predicted instances from the epoch
            pred = np.reshape(pred, (-1, 1)).squeeze()
            faults = np.where(np.not_equal(np.argmax(y_test, 1), pred))[0]
            # Get a sample of size 25 from them
            n = 25
            samples = np.random.choice(faults, n)
            sqrt_n = int(n ** 0.5)

            if dataset_choice == 0:
                # Plot 25 misclassified examples from the Test set MNIST

                plt.figure(figsize=(11, 13))

                cnt = 0
                for i in samples:
                    cnt += 1
                    plt.subplot(sqrt_n, sqrt_n, cnt)
                    plt.subplot(sqrt_n, sqrt_n, cnt).axis('off')
                    # Reshape instance to a 32 by 32 by 1 (Grayscale) image and plot it
                    plt.imshow(X_test[i, 1:].reshape(28, 28) * 255, cmap='gray')
                    # Print the true label of each instance and the false prediction
                    plt.title("True: " + str(np.argmax(y_test, 1)[i]) + "\n Predicted: " + str(pred[i]))

                plt.show()
            else:
                # plot n misclassified examples from the Test set CIFAR-10

                plt.figure(figsize=(11, 13))

                cnt = 0
                for i in samples:
                    cnt += 1
                    plt.subplot(sqrt_n, sqrt_n, cnt)
                    plt.subplot(sqrt_n, sqrt_n, cnt).axis('off')
                    # Reshape instance to a 32 by 32 by 3 (RGB) image and plot it
                    X_test_r = X_test[i][1:][:1024].reshape(32, 32)
                    X_test_g = X_test[i][1:][1024:2048].reshape(32, 32)
                    X_test_b = X_test[i][1:][2048:].reshape(32, 32)
                    X_test_ = np.stack((X_test_r, X_test_g, X_test_b), axis=2)
                    plt.imshow(X_test_)
                    # Print the true label of each instance and the false prediction
                    plt.title("True: " + str(cifar10_labels[np.argmax(y_test, 1)[i]]) + "\n Predicted: " + str(
                        cifar10_labels[pred[i]]))

                plt.show()

    """ Plot the learning graphs for train and test set """

    if plot_learning_graphs:
        # Show learning rate graph
        y = costs_train
        x = np.linspace(0, len(y), len(y))
        plt.plot(x, y, '-', label='train cost')

        y2 = costs_test
        x2 = np.linspace(0, len(y2), len(y2))
        plt.plot(x2, y2, '-', label='test cost')

        plt.ylabel('Average Cost')
        plt.xlabel('Epoch\n\nLast Epoch: Avg_Train_Cost=%g -- Avg_Test_Cost=%g' % (costs_train[len(costs_train) - 1],
                                                                                   costs_test[len(costs_test) - 1]))
        plt.title(
            '%s\nHidden Size=%d -- Batch Size=%d -- Learning rate=%g\nLamda=%g -- Activation=%s -- Weight_Init=%s\n'
            % (('MNIST' if dataset_choice == 0 else 'CIFAR-10'), hidden_size, batch_size, learning_rate, lam_reg,
               activation.__name__, weight_init_mode))
        plt.legend()
        plt.tight_layout()
        if save_graph_plot:
            plt.savefig('./tests_final%d/testrun_%d/testrun_%d.png' % (testhyp, testrun, testrun))
        else:
            plt.show()
        plt.close()

    print('Model results for last epoch:\n')
    print('Avg Train Cost: %f -- Avg Train Accuracy: %f' % (
    costs_train[len(costs_train) - 1], accus_train[len(accus_train) - 1]))
    print('Avg Test Cost: %f -- Avg Test Accuracy: %f\n' % (
        costs_test[len(costs_test) - 1], accus_test[len(accus_test) - 1]))

    return costs_train, accus_train, costs_test, accus_test


""" Main code """

""" Isolated Testing """

costs_train, accus_train, costs_test, accus_test = run_model(dataset_choice=0, epochs=20, hidden_size=100,
                                                             batch_size=100, learning_rate=0.0001, lam_reg=0.01,
                                                             activation=cos, shuffle_data_first=True,
                                                             per_epoch_shuffle_interval=0, per_batch_print=0,
                                                             weight_init_mode='normal',
                                                             plot_false_predicted_interval=20, converge_thres=0.1)
print('Average train cost per epoch:')
print(costs_train)
print('Average train accuracy per epoch:')
print(accus_train)
print('Average test cost per epoch:')
print(costs_test)
print('Average test accuracy per epoch:')
print(accus_test)

exit(1)

""" Testing different parameters and hyperparameters """

testhyp = 1
for lr in [0.001, 0.0001]:
    for lam_r in [0.1, 0.01]:

        testrun = 0

        dataset_c = [0, 1]
        eps = 2
        hid_size = [100, 200, 300]
        b_size = [100]
        act = [softplus, tanh, cos]

        number_of_tests = len(dataset_c) * len(hid_size) * len(b_size) * len(act)
        remaining = 0
        for d in dataset_c:
            # 0 for mnist
            # 1 for cifar-10
            if d == 0:
                X_train, X_test, y_train, y_test = load_data_mnist()  # load_data_cifar10() or load_data_mnist()
                data = (X_train, X_test, y_train, y_test)
            else:
                X_train, X_test, y_train, y_test = load_data_cifar10()  # load_data_cifar10() or load_data_mnist()
                tmp = unpickle('data/cifar-10-batches-py/batches.meta')
                cifar10_labels = [e.decode('utf-8') for e in tmp[b'label_names']]
                data = (X_train, X_test, y_train, y_test, cifar10_labels)
            for a in act:
                for bsize in b_size:
                    for h in hid_size:
                        if not os.path.exists('./tests_final%d/testrun_%d/' % (testhyp, testrun)):
                            os.makedirs('./tests_final%d/testrun_%d/' % (testhyp, testrun))

                        # Start measuring time
                        start = time.time()

                        print('-'*30)

                        # Run the model with the specific parameters and hyperparameters
                        costs_train, accus_train, costs_test, accus_test = run_model(dataset=data, hidden_size=h,
                                                                                     batch_size=bsize,
                                                                                     activation=a, learning_rate=lr,
                                                                                     lam_reg=lam_r, per_batch_print=0,
                                                                                     epochs=eps, save_graph_plot=True)

                        # Write the log file
                        with open('./tests_final%d/testrun_%d/testrun_%d.log' % (testhyp, testrun, testrun), 'w') as wfile:
                            wfile.write('Dataset: %s -- Hidden Size: %d -- Batch Size: %d\n' % (
                                ('MNIST' if d == 0 else 'CIFAR-10'), h, bsize))
                            wfile.write('Activation: %s -- Learning Rate: %g -- Lambda: %g\n' % (a.__name__, lr, lam_r))
                            wfile.write('Epochs: %d\n\n' % eps)
                            wfile.write('Average train cost per epoch:\n')
                            wfile.write(str(costs_train))
                            wfile.write('\nAverage train accuracy per epoch:\n')
                            wfile.write(str(accus_train))
                            wfile.write('\nAverage test cost per epoch:\n')
                            wfile.write(str(costs_test))
                            wfile.write('\nAverage test accuracy per epoch:\n')
                            wfile.write(str(accus_test))

                        # Stop measuring time
                        end = time.time()

                        # Find average remainig time estimate
                        remaining = (remaining + (end - start)) / (testrun+1)

                        print('Remaining time: %g secs' % (remaining * (number_of_tests-testrun)))

                        testrun += 1
        testhyp += 1
