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

    #training part
    for epoch in range(1, epochs):
        error = 0
        for i, x in enumerate(X):
            #misclassification
            if (Y[i]*np.dot(X[i], w)) < 1:
                #missclassified update for weights
                w = w + eta * ( (X[i] * Y[i]) + (-2 * (1/epoch) * w) )
            else:
                #correct classification, update our weights
                w = w + eta * (-2 * (1/epoch) * w)

        errors.append(error)
    #plot the rate of classification errors during training
    plt.plot(errors, '|')
    plt.ylim(0.5,1.5)
    plt.axes().set_yticklabels([])
    plt.xlabel('Epoch')
    plt.ylabel('Missclassified')
    plt.show()

    return w

svm_sgd_plot(X,Y)

for d, sample in enumerate(X):
    # plot the negative samples
    if d < 2:
        plt.scatter(sample[0], sample[1], s=120, marker='_', linewidths= 2)
    # plot the positive samples
    else:
        plt.scatter(sample[0], sample[1], s=120, marker='+', linewidths= 2)
# Ass our test samples
plt.scatter(2,2, s=120, marker='_', linewidths=2, color='yellow')
plt.scatter(4,3, s=120, marker='+', linewidths=2, color='blue')

#print the hyperplane calculated by svm_sgd()
x2 = [w[0], w[1], -w[1], w[0]]
x3 = [w[0], w[1], w[1], -w[0]]

x2x3 = np.array([x2,x3])
X,Y,U,V = zip(*x2x3)
ax = plt.gca()
ax.quiver(X,Y,U,V, scale=1, color='blue')