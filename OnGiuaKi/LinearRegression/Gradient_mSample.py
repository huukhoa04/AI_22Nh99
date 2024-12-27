import numpy as np
from numpy import genfromtxt
import matplotlib.pyplot as plt

data = genfromtxt('data.csv', delimiter=',')
areas = data[:,0]
prices = data[:,1]
data_size = areas.size





data = np.c_[areas, np.ones((data_size, 1))]

lr = 0.01

theta = np.array([[-0.34], [0.04]])

# for debug
epoch_max = 10

#mini patch size
m = 2

for epoch in range(epoch_max):
    for j in range(0, data_size, m):
        # some variables
        sum_of_losses = 0
        gradients = np.zeros((2,))

        for index in range(j, j+m):
            # get mini-batch
            x_i = data[index]
            y_i = prices[index]

            # predict y_hat_i
            y_hat_i = x_i.dot(theta)

            # compute loss
            l_i = (y_hat_i - y_i)*(y_hat_i - y_i)

            # compute gradient
            gradient_i = x_i*2*(y_hat_i - y_i)

            # accumulate gradients
            gradients = gradients + gradient_i
            sum_of_losses = sum_of_losses + l_i

        # normalize
        sum_of_losses = sum_of_losses / m
        gradients = gradients / m







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