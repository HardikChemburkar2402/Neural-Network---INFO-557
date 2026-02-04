"""
The main code for the back propagation assignment. See README.md for details.
"""
import math
from typing import List

import numpy as np
#### ADDITIONAL IMPORTS HERE, IF DESIRED ####


# NOTE: In the docstrings, "UDL" refers to the book Pierce (2023),
#       "Understanding Deep learning".


class SimpleNetwork:
    """A simple feedforward network where all units have sigmoid activation,
    including at final layer (output) of model, and there are no bias term
    parameters, only layer weights. Input, output and weight matrices follow
    denominator layout format (same as UDL).
    """

    @classmethod
    def random(cls, *layer_units: int):
        """Creates a feedforward neural network with the given number of units
        for each layer (including input (first) and output (last) layers).

        Example: create a network, net, with input layer of 3 units, a first
        hidden layer with 4 hidden units, a second hidden layer with 5 hidden
        units, and an output layer with 2 units:
            net = SimpleNetwork.random(3, 4, 5, 2)

        :param layer_units: Number of units for each layer
        :return: the neural network
        """

        # Define a function to initialize weights with uniform distribution
        def uniform(n_in, n_out):
            # Calculate epsilon based on the number of input and output units
            epsilon = math.sqrt(6) / math.sqrt(n_in + n_out)
            # Return a matrix of weights with values uniformly distributed
            # between -epsilon and +epsilon
            return np.random.uniform(-epsilon, +epsilon, size=(n_out, n_in))
        # Create pairs of consecutive layer units
        pairs = zip(layer_units, layer_units[1:])
        # Initialize the network with random weights for each layer
        return cls(*[uniform(i, o) for i, o in pairs])

    def __init__(self, *layer_weights: np.ndarray):
        """Creates a neural network from a list of weight matrices.
        The weights specify linear transformations from one layer to the next, so
        the number of layers is equal to one more than the number of layer_weights
        weight matrices.

        :param layer_weights: A list of weight matrices
        """
        # Store the weight matrices for each layer
        self.layer_weights = layer_weights
        # Calculate the number of layers in the network
        self.num_layers = len(layer_weights) + 1

    def predict(self, input_matrix: np.ndarray) -> np.ndarray:
        """Performs forward propagation over the neural network starting with
        the given input matrix.

        Each unit's output should be calculated by taking a weighted sum of its
        inputs (using the appropriate weight matrix) and passing the result of
        that sum through a (logistic) sigmoid activation function. This includes
        at the final layer of the network.

        (This network does not include bias parameters.)

        :param input_matrix: The matrix of inputs to the network, where each
        column in the matrix represents an input instance for which the neural
        network should make a prediction.
        :return: A matrix of predictions, where each column is the predicted
        outputs - each in the value range (0, 1) - for the corresponding column
        in the input matrix.
        """
        def sigmoid(x):
            # Sigmoid activation function to introduce non-linearity
            return 1 / (1 + np.exp(-x))
        # Initialize activation with input matrix
        activation = input_matrix
        # Forward propagate through each layer
        for weight in self.layer_weights:
            # Compute the weighted sum and apply sigmoid activation
            activation = sigmoid(np.dot(weight, activation))
        # Return the final output after passing through all layers
        return activation

    def predict_zero_one(self, input_matrix: np.ndarray) -> np.ndarray:
        """Performs forward propagation over the neural network starting with
        the given input matrix, and converts the outputs to binary (0 or 1).

        Outputs will be converted to 0 if they are less than 0.5, and converted
        to 1 otherwise.

        :param input_matrix: The matrix of inputs to the network, where each
        column in the matrix represents an instance for which the neural network
        should make a prediction.
        :return: A matrix of predictions, where each column is the predicted
        outputs - each either 0 or 1 - for the corresponding column in the input
        matrix.
        """
        def sigmoid(x):
            # Sigmoid activation function to introduce non-linearity
            return 1 / (1 + np.exp(-x))
        # Initialize activation with input matrix
        activation = input_matrix
        # Forward propagate through each layer
        for weight in self.layer_weights:
            # Compute the weighted sum and apply sigmoid activation
            activation = sigmoid(np.dot(weight, activation))
        # Convert outputs to binary (0 or 1) based on threshold 0.5
        return (activation >= 0.5).astype(int)

    def gradients(self,
                  input_matrix: np.ndarray,
                  target_output_matrix: np.ndarray) -> List[np.ndarray]:
        """Performs backpropagation to calculate the gradients (derivatives of
        the loss with respect to the model weight parameters) for each of the
        weight matrices.

        This method first performs a pass of forward propagation through the
        network, where the pre-activations (f) and activations (h) of each
        layer are stored. (NOTE: this bookkeeping could be performed in
        self.predict(), if desired.)
        This method then applies the following procedure to calculate the
        gradients.

        In the following description, × is matrix multiplication, ⊙ is
        element-wise product, and ⊤ is matrix transpose. The acronym 'w.r.t.'
        is shorhand for "with respect to".

        First, calculate the derivative of the squared loss w.r.t. model's
        final layer, K, activations, Sig[f_K], and the target output matrix, y:

            dl_df[K] = (Sig[f_K] - y)^2

        Then for each layer k in the network, starting with the layer before
        the output layer and working back to the first layer (the input matrix),
        calculate the gradient for the corresponding weight matrix as follows.

        (1) Calculate the derivatives of the loss w.r.t. the weights at the
        layer, dl_dweights[layer] (i.e., the parameter gradients), using the
        derivative of the loss w.r.t. layer pre-activation, dl_df[layer], and
        the activation, h[layer].
        (UDL equation 7.22)
        NOTE: With multiple inputs, there will be one gradient per input
        instance, and these must be summed (element-wise across gradient per
        input) and the resulting summed gradient must be (element-wise) divided
        by the number of input instances. As discussed in class, the simultaneous
        outer product and sum across gradients can be achieved using numpy.matmul,
        leaving only the element-wise division by the number of input instances.
        NOTE: The gradient() method returns the list of gradients per layer,
        so you will need to store the computed gradient per layer in a List
        for return at the end. The order of the gradients should be in
        "forward" order (layer 0 first, layer 1 second, etc...).

        (2) Calculate the derivatives of the loss w.r.t. the activations,
        dl_dh[layer], from the transpose of the weights, weights[layer].⊤,
        and the derivatives of the next pre-activation, dl_df[layer].
        (the second part of the last line of UDL equation 7.24)

        (3) If the current layer is not the 0'th layer, then:
        Calculate the derivatives of the loss w.r.t. the pre-activation
        for the previous layer, dl_df[layer - 1]. This involves the derivative
        of the activation function (sigmoid), dh_df.
        (first part of the last line of UDL eq 7.24)

        :param input_matrix: The matrix of inputs to the network, where each
        column in the matrix represents an instance for which the neural network
        should make a prediction.
        :param target_output_matrix: A matrix of expected outputs, where each column
        is the expected outputs - each either 0 or 1 - for the corresponding column
        in the input matrix.
        :return: List of the gradient matrices, 1 for each weight matrix, in same
        order as the weight matrix layers of the network, from input to output.
        """
        def sigmoid(x):
            # Sigmoid activation function to introduce non-linearity
            return 1 / (1 + np.exp(-x))

        def sigmoid_derivative(x):
            # Derivative of the sigmoid function for backpropagation
            sig = sigmoid(x)
            return sig * (1 - sig)

        # Forward pass: store activations and pre-activations
        activations = [input_matrix]
        pre_activations = []
        activation = input_matrix

        for weight in self.layer_weights:
            # Calculate pre-activation (weighted sum)
            pre_activation = np.dot(weight, activation)
            pre_activations.append(pre_activation)
            # Apply sigmoid activation function
            activation = sigmoid(pre_activation)
            activations.append(activation)

        # Backward pass: calculate gradients
        gradients = []
        # Derivative of loss w.r.t. final layer pre-activation
        dl_df = 2 * (activations[-1] - target_output_matrix)
        dl_df *= sigmoid_derivative(pre_activations[-1])
        for layer in reversed(range(len(self.layer_weights))):
            # Calculate gradient of loss w.r.t. weights
            dl_dweights = np.matmul(dl_df, activations[layer].T) / input_matrix.shape[1]
            gradients.insert(0, dl_dweights)

            if layer > 0:
                # Calculate derivative of loss w.r.t. previous layer's activation
                dl_dh = np.dot(self.layer_weights[layer].T, dl_df)
                # Calculate derivative of loss w.r.t. previous layer's pre-activation
                dl_df = dl_dh * sigmoid_derivative(pre_activations[layer - 1])

        return gradients

    def train(self,
              input_matrix: np.ndarray,
              target_output_matrix: np.ndarray,
              iterations: int = 10,
              learning_rate: float = 0.1) -> None:
        """Trains the neural network on an input matrix and an expected output
        matrix.

        Training should repeatedly (`iterations` times) calculate the gradients,
        and update the model by subtracting the learning rate times the
        gradients from the model weight matrices.

        :param input_matrix: The matrix of inputs to the network, where each
        column in the matrix represents an input instance for which the neural
        network should make a prediction
        :param target_output_matrix: A matrix of expected outputs, where each
        column is the expected outputs - each either 0 or 1 - for the corresponding row in
        the input instance in the input matrix.
        :param iterations: The number of gradient descent steps to take.
        :param learning_rate: The size of gradient descent steps to take, a
        number that the gradients should be multiplied by before updating the
        model weights.
        """
        # Iterate for the given number of iterations
        for _ in range(iterations):
            # Calculate the gradients using backpropagation
            gradients = self.gradients(input_matrix, target_output_matrix)
            # Update the weights by subtracting the learning rate times the gradients
            self.layer_weights = [
            weight - learning_rate * gradient
            for weight, gradient in zip(self.layer_weights, gradients)
            ]
