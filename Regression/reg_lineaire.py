import numpy as np
from sklearn.datasets import make_regression
import matplotlib.pyplot as plt

# Définis une graine (seed) pour la reproductibilité
np.random.seed(42)

def main():
    x, y = make_regression(n_samples=100, n_features=1, noise=10)
    y = y.reshape(100, 1)


    X = np.hstack((x, np.ones((x.shape))))
    theta = np.zeros((2, 1))  # Initialisation avec des zéros

    # Machine learning, choix des paramètres pour l'algo et obtention d'un theta final (a, b) le plus "optimal"
    theta_final = gradient_descent(X, y, theta, learning_rate=0.01, n_iterations=100)
    print(theta_final)

    predictions = model(X, theta_final)
    plt.scatter(x, y)
    plt.plot(x, predictions, c='r')
    plt.show()

# Model
def model(X, theta):
    return X.dot(theta)

# Gradients et Descente de Gradient
def grad(X, y, theta):
    m = len(y)
    return 1/m * X.T.dot(model(X,theta) - y)

def gradient_descent(X, y, theta, learning_rate, n_iterations):
    for i in range(n_iterations):
        theta = theta - learning_rate * grad(X, y, theta)
    return theta

if __name__ == '__main__':
    main()