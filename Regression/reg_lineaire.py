import numpy as np
from sklearn.datasets import make_regression
import matplotlib.pyplot as plt

# Définis une graine (seed) pour la reproductibilité
np.random.seed(42)

n_iterations = 4000 #on choisis le nombre d'iteration ici 

def main():
    x, y = make_regression(n_samples=100, n_features=1, noise=10)
    y = y.reshape(100, 1)


    X = np.hstack((x, np.ones((x.shape))))
    theta = np.zeros((2, 1))  # Initialisation avec des zéros

    # Machine learning, choix des paramètres pour l'algo et obtention d'un theta final (a, b) le plus "optimal"
    theta_final, cost_history = gradient_descent(X, y, theta, learning_rate=0.001, n_iterations=n_iterations)
    print(theta_final)

    predictions = model(X, theta_final)
    plt.scatter(x, y)
    plt.plot(x, predictions, c='r')
    #plt.plot(range(n_iterations), cost_history) #Graoh de la performance de Gradient Descent
    plt.show()

# Model
def model(X, theta):
    return X.dot(theta)

def cost_function(X, y, theta):
    m = len(y)
    return 1/(2*m) * np.sum((model(X,theta) - y) ** 2)

# Gradients et Descente de Gradient
def grad(X, y, theta):
    m = len(y)
    return 1/m * X.T.dot(model(X,theta) - y)

def gradient_descent(X, y, theta, learning_rate, n_iterations):
    cost_history = np.zeros(n_iterations)
    for i in range(n_iterations):
        theta = theta - learning_rate * grad(X, y, theta)
        cost_history[i] = cost_function(X, y, theta) #Cela permet d'avoir un historique des couts pour savoir si lalgo a fait trop ou pas assez diterations
    return theta, cost_history

if __name__ == '__main__':
    main()