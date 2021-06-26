import unittest
import numpy as np
import pickle
import os

class Neuron:
    def __init__(self, weights=[], inp=0, out=0, bias=0):
        self.weights = weights
        self.inp = inp
        self.out = out
        self.bias = bias
        self. error = 0


class Layer:
    def __init__(self, neurons):
        self.neurons = neurons

class NeuralNetwork:
    """Implement/make changes to places in the code that contains #TODO."""

    def __init__(self, input_dim: int, hidden_layer: bool) -> None:
        """
        Initialize the feed-forward neural network with the given arguments.
        :param input_dim: Number of features in the dataset.
        :param hidden_layer: Whether or not to include a hidden layer.
        :return: None.
        """

        # Number of hidden units if hidden_layer = True.
        self.hidden_units = 25

        self.lr = 1e-2

        self.epochs = 400

        self.x_train, self.y_train = None, None
        self.x_test, self.y_test = None, None


        self.input_dim = input_dim
        self.hidden_layer = hidden_layer
        self.layers=[]



    def load_data(self, file_path: str = os.path.join(os.getcwd(), 'data_breast_cancer.p')) -> None:
        """
        Do not change anything in this method.

        Load data for training and testing the model.
        :param file_path: Path to the file 'data_breast_cancer.p' downloaded from Blackboard. If no arguments is given,
        the method assumes that the file is in the current working directory.

        The data have the following format.
                   (row, column)
        x: shape = (number of examples, number of features)
        y: shape = (number of examples)
        """
        with open(file_path, 'rb') as file:
            data = pickle.load(file)
            self.x_train, self.y_train = data['x_train'], data['y_train']
            self.x_test, self.y_test = data['x_test'], data['y_test']

    def train(self) -> None:
        self.layers = self.init_weights()
        for ep in range(self.epochs):
            for j in range(len(self.x_train)):
                xt = self.x_train[j]
                yt = self.y_train[j]
                
                
                input_neurons=[]
                # Update input layer
                for n in range(self.input_dim):
                    neuron=Neuron(out=xt[n])
                    input_neurons.append(neuron)
                input_layer = Layer(input_neurons)

                if not ((j == 0) and (ep == 0)): self.layers.pop(0)
                self.layers.insert(0, input_layer)
                
                # Forward pass
                for m in range(1, len(self.layers)):
                    for neuron in self.layers[m].neurons:
                        sum=0
                        for k in range(len(neuron.weights)):
                            sum+=neuron.weights[k]*self.layers[m-1].neurons[k].out
                        neuron.inp=sum
                        neuron.out=self.sig(sum+neuron.bias)

                # Calculate error in output layer
                for neuron in self.layers[-1].neurons:
                    neuron.error = self.sig_der(neuron.inp+neuron.bias)*(yt - neuron.out)

                # Calculate error in hidden layer
                for m in range(len(self.layers)-2, 1, -1):
                    for neuron in self.layers[m].neurons:
                        sum=0
                        for k in range(len(neuron.weights)):
                            sum+=neuron.weights[k]*self.layers[m+1].neurons[k].error
                        neuron.error = self.sig_der(neuron.inp+neuron.bias)*sum

                # Update network weights 
                for m in range(1, len(self.layers)):
                    for neuron in self.layers[m].neurons:
                        for k in range(len(neuron.weights)):
                            neuron.weights[k]+=self.lr*neuron.error*self.layers[m-1].neurons[k].out
                        neuron.bias+=self.lr*neuron.error
             
    def init_weights(self):
        # Init hidden layer 
        hidden_layer=None
        if self.hidden_layer:
            hidden_neurons = []
            for i in range(self.hidden_units):
                weights = np.random.uniform(-0.5, 0.5, self.input_dim)
                hidden_neurons.append(Neuron(weights=weights))
            hidden_layer = Layer(hidden_neurons)

        # Init output layer
        out_neurons = []
        if self.hidden_layer: 
            weights = np.random.uniform(-0.5, 0.5, self.hidden_units)
        else:
            weights = np.random.uniform(-0.5, 0.5, self.input_dim)
        out_neurons.append(Neuron(weights=weights))
        out_layer = Layer(out_neurons)

        if self.hidden_layer: 
            return [hidden_layer, out_layer]
        else: 
            return [out_layer]


    def sig(self, inp):
        return 1/(1 + np.exp(-inp))

    def sig_der(self, inp):
        f = 1/(1+np.exp(-inp))
        return f * (1 - f)

    def predict(self, x: np.ndarray) -> float:
        """
        Given an example x we want to predict its class probability.
        For example, for the breast cancer dataset we want to get the probability for cancer given the example x.
        :param x: A single example (vector) with shape = (number of features)
        :return: A float specifying probability which is bounded [0, 1].
        """

        # TODO: Implement the forward pass.
        input_neurons=[]
        for n in range(self.input_dim):
            neuron=Neuron(out=x[n])
            input_neurons.append(neuron)
        input_layer = Layer(input_neurons)
        self.layers.pop(0)
        self.layers.insert(0, input_layer)

        for m in range(1, len(self.layers)):
            for neuron in self.layers[m].neurons:
                sum=0
                for k in range(len(neuron.weights)):
                    sum+=neuron.weights[k]*self.layers[m-1].neurons[k].out
                neuron.inp=sum
                neuron.out=self.sig(sum+neuron.bias)
        return self.layers[-1].neurons[0].out        


class TestAssignment5(unittest.TestCase):
    """
    Do not change anything in this test class.

    --- PLEASE READ ---
    Run the unit tests to test the correctness of your implementation.
    This unit test is provided for you to check whether this delivery adheres to the assignment instructions
    and whether the implementation is likely correct or not.
    If the unit tests fail, then the assignment is not correctly implemented.
    """

    def setUp(self) -> None:
        self.threshold = 0.8
        self.nn_class = NeuralNetwork
        self.n_features = 30

    def get_accuracy(self) -> float:
        """Calculate classification accuracy on the test dataset."""
        self.network.load_data()
        self.network.train()

        n = len(self.network.y_test)
        correct = 0
        for i in range(n):
            # Predict by running forward pass through the neural network
            pred = self.network.predict(self.network.x_test[i])
            # Sanity check of the prediction
            assert 0 <= pred <= 1, 'The prediction needs to be in [0, 1] range.'
            # Check if right class is predicted
            correct += self.network.y_test[i] == round(float(pred))
        return round(correct / n, 3)

    def test_perceptron(self) -> None:
        """Run this method to see if Part 1 is implemented correctly."""

        self.network = self.nn_class(self.n_features, False)
        accuracy = self.get_accuracy()
        print(accuracy)
        self.assertTrue(accuracy > self.threshold,
                        'This implementation is most likely wrong since '
                        f'the accuracy ({accuracy}) is less than {self.threshold}.')

    def test_one_hidden(self) -> None:
        """Run this method to see if Part 2 is implemented correctly."""

        self.network = self.nn_class(self.n_features, True)
        accuracy = self.get_accuracy()
        print(accuracy)
        self.assertTrue(accuracy > self.threshold,
                        'This implementation is most likely wrong since '
                        f'the accuracy ({accuracy}) is less than {self.threshold}.')


if __name__ == '__main__':
    unittest.main()
