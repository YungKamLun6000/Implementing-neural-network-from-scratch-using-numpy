import numpy as np

class Layer:
    def __init__(self, input_size, output_size):
        self.weights = np.empty((output_size, input_size), float)
        self.biases = np.empty((output_size, 1), float)

        for j in range(output_size):
            self.weights[j] = np.random.uniform(-1, 1, input_size)
            self.biases[j] = np.random.uniform(-1, 1)

    def forward(self, input):
        output = np.dot(self.weights, input) + self.biases
        return output

    @staticmethod
    def RelU(input, d=False):
        if d:
            return (input > 0).astype(float)
        return np.maximum(0, input)

    @staticmethod
    def Sigmoid(input, d=False):
        if d:
            s = 1 / (1 + np.exp(-input))
            return s*(1-s)
        return 1 / (1 + np.exp(-input))

    @staticmethod
    def Binary_cross_entropy(y_true, y_pred, d=False):
        if d:
           DL = np.mean(y_pred - y_true)
           return DL
        loss = -np.mean(y_true * np.log(y_pred) + (1 - y_true) * np.log(1 - y_pred))
        return loss

    @staticmethod
    def backpropagation(pre_output, z, current_Layer, future_Layer, future_delta, lr):
        da_dz = Layer.RelU(z, d=True)
        delta = np.dot(future_Layer.weights.T, future_delta) * da_dz
        current_Layer.weights -= lr * np.dot(delta, pre_output.T)
        current_Layer.biases -= lr * delta

        return delta