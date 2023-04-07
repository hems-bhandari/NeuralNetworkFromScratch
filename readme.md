Here, I defined a NeuralNetwork class with an __init__ method to initialize the weights of the two layers, a sigmoid method to define the sigmoid activation function, a sigmoid_derivative method to calculate the derivative of the sigmoid function, a feedforward method to calculate the output of the network for a given input, a backpropagation method to propagate the error backwards through the network and adjust the weights, a train method to train the network on a given dataset for a specified number of epochs, and a predict method to make a prediction for a new input.

I then defined a sample dataset for the neural network, created an instance of the NeuralNetwork class with 2 input neurons, 5 hidden neurons, and 1 output neuron, and trained the network on the dataset for 10,000 epochs using a learning rate of 0.1.

Finally, make a prediction for a new input of [1, 0] and print the result.