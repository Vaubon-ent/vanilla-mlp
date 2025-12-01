from services.neural_network import NeuralNetwork

if __name__ == "__main__":
    nn = NeuralNetwork()
    nn.run("TRAINING")
    nn.run("TESTING")