import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

class LabelEncoder:
    def __init__(self):
        self.classes_ = None
        
    def fit_transform(self, y):
        self.classes_ = np.unique(y)
        return np.array([np.where(self.classes_ == label)[0][0] for label in y])
        
    def inverse_transform(self, y):
        return np.array([self.classes_[i] for i in y])

def train_test_split(X, y, test_size=0.2, random_state=None):
    if random_state is not None:
        np.random.seed(random_state)
    n_samples = len(X)
    n_test = int(n_samples * test_size)
    indices = np.random.permutation(n_samples)
    test_indices = indices[:n_test]
    train_indices = indices[n_test:]
    return X[train_indices], X[test_indices], y[train_indices], y[test_indices]

class LogisticRegression:
    def __init__(self, learning_rate=0.01, n_iterations=1000, multi_class=False):
        self.learning_rate = learning_rate
        self.n_iterations = n_iterations
        self.weights = None
        self.bias = None
        self.multi_class = multi_class
        self.losses = []
        
    def sigmoid(self, z):
        return 1 / (1 + np.exp(-z))
    
    def softmax(self, z):
        exp = np.exp(z - np.max(z, axis=1, keepdims=True))
        return exp / np.sum(exp, axis=1, keepdims=True)
    
    def calculate_loss(self, y_true, y_pred):
        if self.multi_class:
            return -np.mean(np.sum(y_true * np.log(y_pred + 1e-15), axis=1))
        return -np.mean(y_true * np.log(y_pred + 1e-15) + (1 - y_true) * np.log(1 - y_pred + 1e-15))
    
    def fit(self, X, y):
        n_samples, n_features = X.shape
        if self.multi_class:
            n_classes = len(np.unique(y))
            self.weights = np.zeros((n_features, n_classes))
            self.bias = np.zeros(n_classes)
            y_onehot = np.eye(n_classes)[y]
        else:
            self.weights = np.zeros(n_features)
            self.bias = 0
            
        for i in range(self.n_iterations):
            if self.multi_class:
                linear_pred = np.dot(X, self.weights) + self.bias
                predictions = self.softmax(linear_pred)
                loss = self.calculate_loss(y_onehot, predictions)
                
                dw = (1/n_samples) * np.dot(X.T, (predictions - y_onehot))
                db = (1/n_samples) * np.sum(predictions - y_onehot, axis=0)
            else:
                linear_pred = np.dot(X, self.weights) + self.bias
                predictions = self.sigmoid(linear_pred)
                loss = self.calculate_loss(y, predictions)
                
                dw = (1/n_samples) * np.dot(X.T, (predictions - y))
                db = (1/n_samples) * np.sum(predictions - y)
            
            self.weights -= self.learning_rate * dw
            self.bias -= self.learning_rate * db
            self.losses.append(loss)
            
        return self.losses
    
    def predict(self, X):
        if self.multi_class:
            linear_pred = np.dot(X, self.weights) + self.bias
            probas = self.softmax(linear_pred)
            return np.argmax(probas, axis=1)
        else:
            linear_pred = np.dot(X, self.weights) + self.bias
            probas = self.sigmoid(linear_pred)
            return (probas >= 0.5).astype(int)

try:
    # Load and preprocess data
    data = pd.read_csv('output_2.csv')
    X = data.iloc[:, :-1].values
    y = data.iloc[:, -1].values

    # Encode labels
    le = LabelEncoder()
    y = le.fit_transform(y)

    # Split data
    X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42)

    # Train model (multi-class)
    model = LogisticRegression(learning_rate=0.01, n_iterations=1000, multi_class=True)
    losses = model.fit(X_train, y_train)

    # Evaluate
    predictions = model.predict(X_test)
    accuracy = np.mean(predictions == y_test)
    print(f"Accuracy: {accuracy:.2f}")
    print(f"Final Loss: {losses[-1]:.4f}")

    # Plot loss curve
    plt.plot(losses)
    plt.title('Training Loss over Iterations')
    plt.xlabel('Iteration')
    plt.ylabel('Loss')
    plt.show()

except Exception as e:
    print(f"Error: {str(e)}")