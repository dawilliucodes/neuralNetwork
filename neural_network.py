import numpy as np
from neuron import Neuron

class NeuralNetwork:
    def __init__(self):
        weights = np.array([0,1])
        bias = 0
        self.h1 = Neuron(weights, bias)
        self.h2 = Neuron(weights, bias)
        self.o1 = Neuron(weights, bias)

    def feed_forward(self, x):
        out_h1 = self.h1.feed_forward(x)
        out_h2 = self.h2.feed_forward(x)

        out_o1 = self.o1.feed_forward(np.array([out_h1, out_h2]))
        return out_o1
    
network = NeuralNetwork()
x = np.array([2, 3])
print(network.feed_forward(x))