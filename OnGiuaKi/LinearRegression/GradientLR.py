import numpy as np
import matplotlib.pyplot as plt
from numpy import genfromtxt

def load_and_preprocess_data(file_path):
    """Load and preprocess data from CSV file."""
    data = genfromtxt(file_path, delimiter=',', skip_header=1)
    X = data[:, 0:3]
    y = data[:, 3:4]
    return X, y

def normalize_features(X):
    """Normalize features using mean and standard deviation."""
    X_mean = np.mean(X, axis=0)
    X_std = np.std(X, axis=0)
    X_normalized = (X - X_mean) / X_std
    return X_normalized, X_mean, X_std

def add_bias_term(X):
    """Add bias term to feature matrix."""
    return np.c_[np.ones((X.shape[0], 1)), X]

def linear_regression(X_b, y, learning_rate=0.01, epochs=1000):
    """Perform linear regression using gradient descent."""
    theta = np.random.randn(X_b.shape[1], 1)
    losses = []
    for _ in range(epochs):
        y_pred = X_b.dot(theta)
        loss = np.mean((y_pred - y) ** 2)
        losses.append(loss)
        gradients = 2 * X_b.T.dot(y_pred - y) / len(y)
        theta -= learning_rate * gradients
    return theta, losses

def predict_sales(X, X_mean, X_std, theta):
    """Predict sales for given features."""
    X_normalized = (X - X_mean) / X_std
    X_b = add_bias_term(X_normalized)
    return X_b.dot(theta)

def plot_gradient_descent(losses):
    """Plot the gradient descent progress."""
    plt.figure(figsize=(10, 6))
    plt.plot(losses)
    plt.title('Gradient Descent Progress')
    plt.xlabel('Iterations')
    plt.ylabel('Mean Squared Error')
    plt.show()

# Main execution
if __name__ == "__main__":
    # Load and preprocess data
    X, y = load_and_preprocess_data('advertising.csv')
    
    # Normalize features
    X_normalized, X_mean, X_std = normalize_features(X)
    
    # Add bias term
    X_b = add_bias_term(X_normalized)
    
    # Perform linear regression
    theta, losses = linear_regression(X_b, y)
    
    # Plot gradient descent progress
    plot_gradient_descent(losses)
    
    # Select a random sample
    random_index = np.random.randint(0, len(X))
    sample_X = X[random_index]
    sample_y = y[random_index]
    
    print("Original sample:")
    print(f"Features: TV={sample_X[0]:.2f}, Radio={sample_X[1]:.2f}, Newspaper={sample_X[2]:.2f}")
    print(f"Actual Sales: ${sample_y[0]:.2f}")
    
    # Predict sales for the random sample
    predicted_sales = predict_sales(sample_X.reshape(1, -1), X_mean, X_std, theta)
    
    print(f"\nPredicted Sales: ${predicted_sales[0, 0]:.2f}")
    print(f"Difference: ${abs(predicted_sales[0, 0] - sample_y[0]):.2f}")

    # Linear Reversing: Find features that would result in the actual sales
    print("\nLinear Reversing:")
    target_sales = sample_y[0]
    
    # We'll use a simple grid search to find features that approximate the target sales
    best_features = None
    min_difference = float('inf')
    
    for tv in np.linspace(0, 300, 100):
        for radio in np.linspace(0, 50, 50):
            for newspaper in np.linspace(0, 100, 50):
                features = np.array([[tv, radio, newspaper]])
                predicted = predict_sales(features, X_mean, X_std, theta)
                difference = abs(predicted[0, 0] - target_sales)
                
                if difference < min_difference:
                    min_difference = difference
                    best_features = features[0]

    print(f"Reversed Features: TV={best_features[0]:.2f}, Radio={best_features[1]:.2f}, Newspaper={best_features[2]:.2f}")
    print(f"Predicted Sales from Reversed Features: ${predict_sales(best_features.reshape(1, -1), X_mean, X_std, theta)[0, 0]:.2f}")
    print(f"Target Sales: ${target_sales:.2f}")
    print(f"Difference: ${min_difference:.2f}")