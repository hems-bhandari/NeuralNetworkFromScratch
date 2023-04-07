import numpy as np

# Define the neural network class
class NeuralNetwork:
    def __init__(self, input_size, hidden_size, output_size):
        # Initialize the weights for the two layers
        self.weights1 = np.random.rand(input_size, hidden_size)
        self.weights2 = np.random.rand(hidden_size, output_size)

    def sigmoid(self, x):
        # Define the sigmoid activation function
        return 1 / (1 + np.exp(-x))

    def sigmoid_derivative(self, x):
        # Calculate the derivative of the sigmoid activation function
        return x * (1 - x)

    def feedforward(self, X):
        # Calculate the output of the neural network for a given input
        self.layer1 = self.sigmoid(np.dot(X, self.weights1))
        self.layer2 = self.sigmoid(np.dot(self.layer1, self.weights2))
        return self.layer2

    def backpropagation(self, X, y, learning_rate):
        # Propagate the error backwards through the network and adjust the weights
        output_error = y - self.layer2
        output_delta = output_error * self.sigmoid_derivative(self.layer2)

        hidden_error = np.dot(output_delta, self.weights2.T)
        hidden_delta = hidden_error * self.sigmoid_derivative(self.layer1)

        self.weights2 += learning_rate * np.dot(self.layer1.T, output_delta)
        self.weights1 += learning_rate * np.dot(X.T, hidden_delta)

    def train(self, X, y, learning_rate, num_epochs):
        # Train the neural network on a given dataset for a specified number of epochs
        for i in range(num_epochs):
            output = self.feedforward(X)
            self.backpropagation(X, y, learning_rate)

    def predict(self, X):
        # Make a prediction for a given input
        return self.feedforward(X)

# Define a sample dataset for the neural network
X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
y = np.array([[0], [1], [1], [0]])

# Create an instance of the neural network and train it on the dataset
nn = NeuralNetwork(input_size=2, hidden_size=5, output_size=1)
nn.train(X, y, learning_rate=0.1, num_epochs=10000)

# Make a prediction for a new input
new_input = np.array([1, 0])
prediction = nn.predict(new_input)
print(prediction)
