import numpy as np
from numpy import genfromtxt
import matplotlib.pyplot as plt

data = genfromtxt('data.csv', delimiter=',')
areas = data[:,0]
prices = data[:,1]
data_size = areas.size


# forward
def predict(x, theta):
    return x.dot(theta)

# compute gradient
def gradient(y_hat, y, x):
    dtheta = 2*x*(y_hat-y)
    return dtheta

# update weights
def update_weight(theta, lr, dtheta):
    dtheta_new = theta - lr*dtheta
    return dtheta_new

#vector [x, b]
data = np.c_[areas, np.ones((data_size, 1))]
print(data)

#init learning rate
n = 0.01
theta = np.array([[-0.34], [0.04]]) # [w, b]
print('theta', theta)


# number of epochs
epoch_max = 10

for epoch in range(epoch_max):
    for i in range(data_size):
        # get a sample
        x = data[i]
        y = prices[i:i+1]

        # predict y_hat
        y_hat = predict(x, theta)

        # compute loss
        loss = (y_hat-y)*(y_hat-y)

        # compute gradient
        dtheta = gradient(y_hat, y, x)

        # update weights
        theta = update_weight(theta, n, dtheta)

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