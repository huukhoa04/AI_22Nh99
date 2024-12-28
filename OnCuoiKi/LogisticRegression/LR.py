import numpy as np
import matplotlib.pyplot as plt


def prod(w, X):
   return np.dot(w.T, X)


def sigmoid(s):
   return 1 / (1 + np.exp(-s))


def my_logistic_sigmoid_regression(X, y, w_init, eta, epsilon=1e-3, M=10000):
   w = [w_init]
   N = X.shape[1]
   d = X.shape[0]
   count = 0
   check_w_after = 20
   while count < M:
      # mix data
      mix_id = np.random.permutation(N)
      for i in mix_id:
         xi = X[:, i].reshape(d, 1)
         yi = y[i]
         zi = sigmoid(np.dot(w[-1].T, xi))
         w_new = w[-1] + eta * (yi - zi) * xi
         count += 1
         # stopping criteria
         if count % check_w_after == 0:
               if np.linalg.norm(w_new - w[-check_w_after]) < epsilon:
                  return w
         w.append(w_new)
   return w

# method display
def display(w):
   # old
   # X0 = X[0, np.where(y == 0)][0]
   # new
   X0 = X[0, np.where(y == 0)][0]
   y0 = y[np.where(y == 0)]
   # old
   # X1 = X[0, np.where(y == 1)][0]
   X1 = X[0, np.where(y == 1)][0]
   y1 = y[np.where(y == 1)]

   plt.plot(X0, y0, 'ro', markersize=8)
   plt.plot(X1, y1, 'bs', markersize=8)
   xx = np.linspace(0, 6, 1000)
   w0 = w[-1][0][0]
   w1 = w[-1][1][0]
   threshold = -w0 / w1
   yy = sigmoid(w0 + w1 * xx)
   plt.axis([-2, 8, -1, 2])
   plt.plot(xx, yy, 'g-', linewidth=2)
   plt.plot(threshold, .5, 'y^', markersize=8)
   plt.xlabel('studying hours')
   plt.ylabel('predicted probability of pass')
   plt.show()

def load_data(filepath):
   try:
      data = np.loadtxt(filepath, delimiter=',')
      X = data[:, :-1].T  # Features
      y = data[:, -1]     # Labels
      return X, y
   except Exception as e:
      print(f"Error loading data: {e}")
      raise

def visualize_data(X, y, w=None):
   plt.scatter(X[0, y==0], X[1, y==0], c='red', label='Class 0')
   plt.scatter(X[0, y==1], X[1, y==1], c='blue', label='Class 1')
   
   if w is not None:
      # Plot decision boundary
      x1 = np.arange(np.min(X[0]), np.max(X[0]), 0.1)
      x2 = -(w[0] + w[1]*x1)/w[2]
      plt.plot(x1, x2, 'g--', label='Decision Boundary')
   
   plt.xlabel('X1')
   plt.ylabel('X2')
   plt.legend()
   plt.show()
def format_float_array(arr, precision=8):
   """Format float array to fixed precision"""
   return np.array2string(arr, precision=precision, suppress_small=True)
if __name__ == '__main__':
   # print("ok")
   # # data:
   # X = np.array([[0.50, 0.75, 1.00, 1.25, 1.50, 1.75, 1.75, 2.00, 2.25, 2.50,
   #                2.75, 3.00, 3.25, 3.50, 4.00, 4.25, 4.50, 4.75, 5.00, 5.50]])
   # y = np.array([0, 0, 0, 0, 0, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 1, 1, 1, 1, 1])
   # # extended data
   # Xbar = np.concatenate((np.ones((1, X.shape[1])), X), axis=0)
   # epsilon = .05
   # d = Xbar.shape[0]
   # w_init = np.random.randn(d, 1)
   # w = my_logistic_sigmoid_regression(Xbar, y, w_init, epsilon)
   # print(w[-1].T)
   # print(sigmoid(np.dot(w[-1].T, Xbar)))
   # display(w)



   # Load data
   X, y = load_data('output_2.csv')
   
   # Generate random input data
   # np.random.seed(727)  # For reproducibility
   
   N = 50  # Number of samples
   X = np.random.randn(2, N)  # 2 features
   # Generate labels using a simple rule
   y = (X[0] + X[1] > 0).astype(int)

   # Add bias term
   N = X.shape[1]
   Xbar = np.vstack((np.ones((1, N)), X))
   
   # Initialize parameters
   w_init = np.random.randn(Xbar.shape[0], 1)
   
   # Train model
   w = my_logistic_sigmoid_regression(Xbar, y, w_init, eta=0.05)
   # Print final weights and predicted probabilities
   print("Final weights:", w[-1].T)
   print("Predicted probabilities:", format_float_array(sigmoid(np.dot(w[-1].T, Xbar))))
   # Visualize results
   visualize_data(X, y, w[-1])
