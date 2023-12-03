import numpy as np
from sklearn.datasets import make_regression
import matplotlib.pyplot as plt

# Définis une graine (seed) pour la reproductibilité
np.random.seed(5336)

n_iterations = 10 #on choisis le nombre d'iteration ici 
learning = 0.5



theta = np.zeros((2, 1))  


def main():
    x, y = make_regression(n_samples=100, n_features=1, noise=10) #on choisi une plage de donnée (100) un seul feature xi et un bruit de fond de 10 (noise)
    y = y.reshape(100, 1)  #on redimmensionne notre vecteur y avec les valeurs de y 1 colonne


    X = np.hstack((x, np.ones((x.shape[0], 1)))) #on place une matrice de one de dimension xshape pour les lignes et 1 pour colonne
    

    # Machine learning, choix des paramètres pour l'algo et obtention d'un theta final (a, b) le plus "optimal"
    theta_final, cost_history = gradient_descent(X, y, theta, learning_rate=learning, n_iterations=n_iterations)
    

    predictions = model(X, theta_final) #on appelle la fonctrion model pour "fusionner" les deux matrices
    print(f"La fonction polynomiale de theta final est  {theta_final[0, 0]:.2f}x + {theta_final[1, 0]:.2f}")
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

def coef_determination(y, pred):
    u = ((y - pred)**2).sum()
    v = ((y - y.mean())**2).sum()
    return 1 - u/v

if __name__ == '__main__':
    main()