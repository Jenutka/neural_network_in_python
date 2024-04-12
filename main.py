import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation


class NeuralNetwork():

    def __init__(self):
        np.random.seed(1)
        self.learning_data = []
        self.synaptic_weights = 2 * np.random.random((3, 1)) - 1

    def sigmoid(self, x):
        return 1 / (1 + np.exp(-x))

    def sigmoid_derivative(self, x):
        return x * (1 - x)

    def train(self, training_inputs, training_outputs, training_iterations):

        for iteration in range(training_iterations):

            output = self.think(training_inputs)
            self.learning_data.append(output)
            error = training_outputs - output
            adjustments = np.dot(training_inputs.T, error \
                                * self.sigmoid_derivative(output))
            self.synaptic_weights += adjustments

    def think(self, inputs):

        inputs = inputs.astype(float)
        output = self.sigmoid(np.dot(inputs, self.synaptic_weights))

        return output


if __name__ == "__main__":

    neural_network = NeuralNetwork()

    print("Random synaptic weights: ")
    print(neural_network.synaptic_weights)

    training_inputs = np.array([[0,0,1],
                                [1,1,1],
                                [1,0,1],
                                [0,1,1]])

    training_outputs = np.array([[0,1,1,0]]).T

    epochs = 1000

    neural_network.train(training_inputs, training_outputs, epochs)

    print("learning data: ")
    print(neural_network.learning_data)

    learning_example = []
    for i in neural_network.learning_data:
        learning_example.append(i[0][0])

    print(learning_example)

    fig, ax = plt.subplots()
    ax.plot(np.arange(0.0, epochs, 1), learning_example)

    plt.show()

    print("Synaptic weights after training: ")
    print(neural_network.synaptic_weights)

    A = str(input("Input 1: "))
    B = str(input("Input 2: "))
    C = str(input("Input 3: "))

    print("New situation: input data = ", A, B, C)
    print("Output data: ")
    print(neural_network.think(np.array([A, B, C])))
