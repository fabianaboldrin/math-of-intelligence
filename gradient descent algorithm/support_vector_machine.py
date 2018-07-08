import numpy as np
from matplotlib import pyplot as plt
# %matplotlib inline

# Defining the data

# Imput data - of the form [X value, Y value, Bias term]

X = np.array([
    [-2,-4,-1],
    [4,1,-1],
    [1,6,-1],
    [2,4,-1],
    [6,2,-1]]
    )

# Associated output labels - First 2 exemples are labeled '-1' and last 3 '1'

y = np.array([-1,-1,1,1,1])

# ploting these exemples on 2D graph

for d, sample in enumerate(X):
    # plot the negative samples
    if d < 2:
        plt.scatter(sample[0], sample[1], s=120, marker='_', linewidths= 2)
    # plot the positive samples
    else:
        plt.scatter(sample[0], sample[1], s=120, marker='+', linewidths= 2)

# print a possible hyperplane, that is separating the two classes.
plt.plot([-2,6], [6,0.5])

# perform a stochastic gradient descent to learn the seperating hyperplan

def svm_sgd_plot(X,Y):
    # inicialize SVMs weight vector with zeros
    w = np.zeros(len(X[0]))
    # the learning rate
    eta = 1
    # how many interations to train for
    epochs = 100000
    #store misclassifications to plot it over time
    errors = []