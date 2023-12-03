import numpy as np
from sklearn.datasets import make_regression
import matplotlib.pyplot as plt

np.random.seed(477)
n_iterations = 500
learning = 0.01


theta = np.zeros((4, 1))   #nouvelle shape pour ajouter c

"""
Fonction cubique
"""


def main():
    x, y = make_regression(n_samples=100, n_features=1, noise=10)
    
    y = np.sqrt(y + abs(y/2))

    
    y = y.reshape(100, 1)  
    X = np.hstack((x, np.ones((x.shape[0], 1))))
    X = np.hstack((x**2, X))   #On ajoute ici une colonne x^2 a gauche dans la matrice X pour avoir une fonction ax^2 +bx + c
    X = np.hstack((x**3, X)) #Pour fonction cubique ax^3 + bx^2 + cx + d

    # Machine learning, choix des paramètres pour l'algo et obtention d'un theta final (a, b) le plus "optimal"
    theta_final, cost_history = gradient_descent(X, y, theta, learning_rate=learning, n_iterations=n_iterations)
    predictions = model(X, theta_final) #on appelle la fonctrion model pour "fusionner" les deux matrices

    
    print(f"La fonction polynomiale de theta final est  {theta_final[0, 0]:.2f}x^3 + {theta_final[1, 0]:.2f}x^2 + {theta_final[2, 0]:.2f}x + {theta_final[3, 0]:.2f}")
    print(f"Le coefficient de determination est de {coef_determination(y, predictions)}") 
    
    
    #Premiere figure
    plt.figure()
    plt.scatter(x, y)
    plt.plot(x, predictions, c='r', label='Fonction theta')
    plt.title('FIgure 1')
    
    #Deuxieme figure
    plt.figure()
    plt.plot(range(n_iterations), cost_history) #Graoh de la performance de Gradient Descent
    plt.title('Figure 2')
    
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

def coef_determination(y, pred):
    u = ((y - pred)**2).sum()
    v = ((y - y.mean())**2).sum()
    return 1 - u/v

if __name__ == '__main__':
    main()