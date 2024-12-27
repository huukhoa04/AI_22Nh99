import numpy as np
from numpy import genfromtxt
import matplotlib.pyplot as plt

data = genfromtxt('data.csv', delimiter=',')
areas = data[:,0]
prices = data[:,1]
data_size = areas.size


data = np.c_[areas, np.ones((data_size, 1))]

n_epochs = 10
lr = 0.01

theta = np.array([[-0.34], [0.04]])

# for debug
losses = []

for epoch in range(n_epochs):
    sum_of_losses = 0
    gradients = np.zeros((2, 1))

    for index in range(data_size):
        # get data
        x_i = data[index:index + 1]
        y_i = prices[index:index + 1]

        # compute output y_hat_i
        y_hat_i = x_i.dot(theta)

        # compute loss
        l_i = (y_hat_i - y_i) * (y_hat_i - y_i)

        # compute gradient
        g_l_i = 2 * (y_hat_i - y_i)
        gradient = x_i.T.dot(g_l_i)

        # accumulate gradient
        gradients = gradients + gradient
        sum_of_losses = sum_of_losses + l_i

    # normalize
    sum_of_losses = sum_of_losses / data_size
    gradients = gradients / data_size

    # for debug
    losses.append(sum_of_losses[0][0])

    # update
    theta = theta - lr * gradients



print(type(areas))
print('areas:', areas)
print('prices:', prices)
print('data_size:', data_size)

plt.scatter(areas, prices)
plt.xlabel('areas')
plt.ylabel('prices')
plt.xlim(3,7)
plt.ylim(4,10)
plt.show()