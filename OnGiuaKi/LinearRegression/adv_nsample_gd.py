import numpy as np
from numpy import genfromtxt
import matplotlib.pyplot as plt

# Load data from advertising.csv
data = genfromtxt('advertising.csv', delimiter=',', skip_header=1)

# Extract features (e.g., TV, Radio, Newspaper) and target (Sales)
# Assuming the columns are in the order: TV, Radio, Newspaper, Sales
features = data[:, :-1]  # All columns except the last one
sales = data[:, -1]      # The last column

data_size = features.shape[0]

# Add a column of ones to the features to account for the bias term
features = np.c_[features, np.ones((data_size, 1))]

# Hyperparameters
n_epochs = 10
lr = 0.01

# Initialize theta with random values
theta = np.random.randn(features.shape[1], 1)

# For debug
losses = []

# Gradient descent loop
for epoch in range(n_epochs):
    sum_of_losses = 0
    gradients = np.zeros((features.shape[1], 1))

    for index in range(data_size):
        # Get data
        x_i = features[index:index + 1]
        y_i = sales[index:index + 1].reshape(-1, 1)

        # Compute output y_hat_i
        y_hat_i = x_i.dot(theta)

        # Compute loss
        l_i = (y_hat_i - y_i) ** 2

        # Compute gradient
        g_l_i = 2 * (y_hat_i - y_i)
        gradient = x_i.T.dot(g_l_i)

        # Accumulate gradient
        gradients = gradients + gradient
        sum_of_losses = sum_of_losses + l_i

    # Normalize
    sum_of_losses = sum_of_losses / data_size
    gradients = gradients / data_size

    # For debug
    losses.append(sum_of_losses[0][0])

    # Update
    theta = theta - lr * gradients

# Print the final theta values
print('Theta:', theta)

# Plot the losses
plt.plot(losses)
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Loss over Epochs')
plt.show()