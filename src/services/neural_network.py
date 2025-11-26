from math import exp
from typing import List
import numpy as np

class NeuralNetwork:

    # Constantes
    # NB_PARAM = None

    layers = []
    weights_mats = []
    bias_mats = []

    def __init__(self):
        # Init neural layer
        self.layers.append(np.zeros((1, 784)))
        self.layers.append(np.zeros((1, 16)))
        self.layers.append(np.zeros((1, 16)))
        self.layers.append(np.zeros((1, 10)))

        # Init weights and bias matrix
        self.weights_mats.append(np.zeros((16, 784)))
        self.weights_mats.append(np.zeros((16, 16)))
        self.weights_mats.append(np.zeros((10, 16)))

        self.bias_mats.append(np.zeros((16, 1)))
        self.bias_mats.append(np.zeros((16, 1)))
        self.bias_mats.append(np.zeros((10, 1)))


    # Calcule la sortie pour une image en entrée donnée
    # param:
    # - entry => layers[0], un vect de dim(1, 784)
    # - weights => weights_mats, l'ensemble des matrice de poids
    # - bias => bias_mats, l'ensemble des matrice de biais
    def forward_prop(self, entry: np.array, weights: List, bias: List):
        # Retourne le vecteur de sortie de dim(1, 10)
        current = entry 
        for w, b in zip(weights, bias):
            current = self.sigmoid(np.dot(current, w) + b)
        return current


    def back_prop(self):
        pass

    def cost(self):
        pass

    def training(self):
        pass

    def sigmoid(self, x):
        return 1 / (1 + np.exp(-x))