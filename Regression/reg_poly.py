import numpy as np
from sklearn.datasets import make_regression
import matplotlib.pyplot as plt

np.random.seed(4478)
n_iterations = 4000 


theta = np.zeros((3, 1))   #nouvelle shape pour ajouter c

"""
nous devons aussi ici changer theta car nous avions besoin que de a et b pour une fonction lineaire or 
pour une fonction polynomiale de degré 2 nous avons besoin d'une troisieme variable c donc theta passe dune shape(2,1) a (3,1)
"""


def main():
    x, y = make_regression(n_samples=100, n_features=1, noise=10)
    
    y = y + abs(y/2) #pour un peu "casser" la courbe pour avoir une allure moins lineaire pour notre dataset
    
    y = y.reshape(100, 1)  
    X = np.hstack((x, np.ones((x.shape[0], 1))))
    X = np.hstack((x**2, X))   #On ajoute ici une colonne x^2 a gauche dans la matrice X pour avoir une fonction ax^2 +bx + c

    theta_final, cost_history = gradient_descent(X, y, theta, learning_rate=0.001, n_iterations=n_iterations)
    print(theta_final)

    predictions = model(X, theta_final) 
    plt.scatter(x, y)
    plt.plot(x, predictions, c='r')
    #plt.plot(range(n_iterations), cost_history) #Graoh de la performance de Gradient Descent
    plt.show()

def model(X, theta):
    return X.dot(theta)

def cost_function(X, y, theta):
    m = len(y)
    return 1/(2*m) * np.sum((model(X,theta) - y) ** 2)

def grad(X, y, theta):
    m = len(y)
    return 1/m * X.T.dot(model(X,theta) - y)

def gradient_descent(X, y, theta, learning_rate, n_iterations):
    cost_history = np.zeros(n_iterations)
    for i in range(n_iterations):
        theta = theta - learning_rate * grad(X, y, theta)
        cost_history[i] = cost_function(X, y, theta) 
    return theta, cost_history

if __name__ == '__main__':
    main()