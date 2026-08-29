from neural_network.layer import Layer
import numpy as np

Layer1 = Layer(2, 2)
Layer2 = Layer(2, 2)
Layer3 = Layer(2, 1)

epoch = 3000
lr = 0.1


X_train = [
    np.array([[1], [1]], dtype=np.float32),
    np.array([[1], [0]], dtype=np.float32),
    np.array([[0], [1]], dtype=np.float32),
    np.array([[0], [0]], dtype=np.float32)
]
y_train = [np.array([[0]]),      np.array([[1]]),      np.array([[1]]),      np.array([[0]])]

for k in range(epoch):
    loss = 0
    for x, y_true in zip(X_train, y_train):
        z1 = Layer1.forward(x)
        a1 = Layer.RelU(z1)
        z2 = Layer2.forward(a1)
        a2 = Layer.RelU(z2)
        z3 = Layer3.forward(a2)
        a3 = Layer.Sigmoid(z3)

        loss += Layer.Binary_cross_entropy(y_true, a3)
        #Generalized Delta Rule
        dL_da3 = Layer.Binary_cross_entropy(y_true, a3, d=True)
        da3_dz3 = Layer.Sigmoid(z3, d=True)
        delta3 = dL_da3 * da3_dz3  # element-wise product

        da2_dz2 = Layer.RelU(z2, d=True)
        delta2 = np.dot(Layer3.weights.T, delta3) * da2_dz2

        da1_dz1 = Layer.RelU(z1, d=True)
        delta1 = np.dot(Layer2.weights.T, delta2) * da1_dz1

        Layer3.weights -= lr * np.dot(delta3, a2.T)
        Layer3.biases -= lr * delta3

        Layer2.weights -= lr * np.dot(delta2, a1.T)
        Layer2.biases -= lr * delta2

        Layer1.weights -= lr * np.dot(delta1, x.T)
        Layer1.biases -= lr * delta1

    print(f"Epoch {k} - Avg Loss: {loss / 4:.4f}")

for x, y_true in zip(X_train, y_train):
    z1 = Layer1.forward(x)
    a1 = Layer.RelU(z1)
    z2 = Layer2.forward(a1)
    a2 = Layer.RelU(z2)
    z3 = Layer3.forward(a2)
    a3 = Layer.Sigmoid(z3)
    print(a3)
