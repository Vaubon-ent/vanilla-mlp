from services.neural_network import NeuralNetwork
import time

if __name__ == "__main__":
    nn = NeuralNetwork()


    print("Démarrage de l'entrainement...")
    start_time = time.time()
    nn.training()
    end_time = time.time()
    print("Entrainement terminé !")
    print(f"Entrainement fini en: {end_time - start_time:.2f} sec")