import numpy as np
import matplotlib.pyplot as plt

def pred(w, x):
    return np.sign(np.dot(w.T, x))

def has_converged(X, y, w):
    return np.array_equal(pred(w, X), y)

def perceptron(X, y, w_init):
    w = [w_init]
    d = X.shape[0]
    mis_points = []
    
    while True:
        # mix data
        mix_id = np.random.permutation(X.shape[1])
        for i in range(X.shape[1]):
            xi = X[:, mix_id[i]].reshape(d, 1)
            yi = y[0, mix_id[i]]
            if pred(w[-1], xi)[0] != yi:
                mis_points.append(mix_id[i])
                w_new = w[-1] + yi * xi
                w.append(w_new)
        
        if has_converged(X, y, w[-1]):
            break
    
    return (w, mis_points)

def load_data(filepath):
    try:
        # Skip header row with skiprows=1
        data = np.loadtxt(filepath, delimiter=',', skiprows=1)
        X = data[:, :-1].T  # All columns except last
        y = data[:, -1].reshape(1, -1)  # Last column
        y = 2 * (y > 0) - 1  # Convert to -1/+1
        
        # Add bias term
        N = X.shape[1]
        Xbar = np.concatenate((np.ones((1, N)), X), axis=0)
        return Xbar, y
    except Exception as e:
        print(f"Error loading data: {e}")
        print("Please ensure CSV format is correct:")
        print("- Skip headers")
        print("- Numbers only")
        print("- Last column contains class labels (0/1)")
        raise


def visualize_perceptron(X, y, w=None):
    plt.figure(figsize=(10, 5))
    
    # Plot training data
    plt.subplot(121)
    pos = np.where(y[0] == 1)[0]
    neg = np.where(y[0] == -1)[0]
    
    plt.scatter(X[1, pos], X[2, pos], c='blue', label='Class 1')
    plt.scatter(X[1, neg], X[2, neg], c='red', label='Class -1')
    
    # Plot decision boundary if w is provided
    if w is not None:
        # Ensure w is properly shaped and get first 3 components
        w = w.flatten()[:3]
        w0, w1, w2 = w
        
        # Calculate decision boundary line
        x1 = np.array([-2, 2])
        x2 = -(w0 + w1*x1)/w2
        plt.plot(x1, x2, 'g--', label='Decision Boundary')
    
    plt.xlabel('X1')
    plt.ylabel('X2')
    plt.legend()
    plt.grid(True)
    plt.title('Perceptron Classification')
    plt.show()


if __name__ == '__main__':

    # means = [[-1, 0], [1, 0]]
    # cov = [[.3, .2], [.2, .3]]
    # N = 10

    # X0 = np.random.multivariate_normal(means[0], cov, N).T
    # X1 = np.random.multivariate_normal(means[1], cov, N).T
    # X = np.concatenate((X0, X1), axis=1)
    # y = np.concatenate((np.ones((1, N)), -1 * np.ones((1, N))), axis=1)
    
    # # Xbar
    # Xbar = np.concatenate((np.ones((1, 2 * N)), X), axis=0)
    # load external data

    Xbar, y = load_data('data.csv')

    visualize_perceptron(Xbar, y)

    w_init = np.random.randn(Xbar.shape[0], 1)
    (w, m) = perceptron(Xbar, y, w_init)

    visualize_perceptron(Xbar, y, w[-1])
    print(w[-1].T)
