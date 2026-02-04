import numpy as np 

def compute_log_loss(X_b, y, theta):
    probs = sigmoid(X_b @ theta)
    eps = 1e-15
    probs = np.clip(probs, eps, 1 - eps)
    loss = -np.mean(y * np.log(probs) + (1 - y) * np.log(1 - probs))
    return loss

def sigmoid(z):
    # Clip to avoid overflow
    z = np.clip(z, -500, 500)
    return 1.0 / (1 + np.exp(-z))

 # X is the matrix with all images and features 
 # y is the vector of labels 
def calculate_gradient(theta, X, y):
    m = y.size 
    return (X.T @ (sigmoid(X @ theta) - y)) / m

def gradient_descent(X, y, alpha=0.1, num_iter = 100, tol = 1e-7):
    # Initialize bias to 1 and concatenate the bias column to the original X matriz of samples x features
    X_b = np.c_[np.ones((X.shape[0],1)), X]
    # Initialize weights to 0, the weights have the same dimension of features ( 1 weight per feature)
    theta = np.zeros(X_b.shape[1])
    
    # y is a column vector
    y = y.reshape(-1)
    loss_history = []

    for i in range(num_iter):
        grad = calculate_gradient(theta, X_b, y)
        theta -= alpha*grad
        loss = compute_log_loss(X_b, y, theta)
        loss_history.append(loss)
        if np.linalg.norm(grad) < tol:
            break 
    return theta, loss_history

def train_ovr(X_train, y_train, num_classes=5, alpha=0.01, num_iter=2000):
    # list to save all the weights for each model
    thetas = []
    losses = []
    for k in range(num_classes):
        # create binary classes for each class (if y_train equal k then put 1 in the vector)
        y_k = (y_train == k).astype(int) 
        # Do gradient descent to each y_k
        theta_k, loss_history_k = gradient_descent(X_train, y_k, alpha=alpha, num_iter=num_iter) 
        # Save the learned parameters to each class, for class 0 row 0 contains the parameters and so on 
        thetas.append(theta_k) 
        losses.append(loss_history_k)
    return np.array(thetas), losses

def predict_multiclass_ovr(X, thetas):
    # X: (N, d)
    # thetas: (K, d+1) because we are adding the column of bias 
    # Add column for bias
    X_b = np.c_[np.ones((X.shape[0], 1)), X]   # (N, d+1)
    # Calculate scores to each class 
    raw_scores = X_b @ thetas.T
    scores = sigmoid(raw_scores)
    # Choose the class with the higher score
    y_pred = np.argmax(scores, axis=1)         
    return y_pred

