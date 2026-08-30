from neural_network.layer import Layer
import numpy as np
import matplotlib.pyplot as plt

Layer1 = Layer(2, 2)
Layer2 = Layer(2, 2)
Layer3 = Layer(2, 1)

epoch = 5000
lr = 0.1


X_train = [
    np.array([[1], [1]], dtype=np.float32),
    np.array([[1], [0]], dtype=np.float32),
    np.array([[0], [1]], dtype=np.float32),
    np.array([[0], [0]], dtype=np.float32)
]
y_train = [np.array([[0]]),      np.array([[1]]),      np.array([[1]]),      np.array([[0]])]

loss_progress = []
epoch_x = range(epoch)

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
        delta3 = dL_da3 * da3_dz3

        Layer3.weights -= lr * np.dot(delta3, a2.T)
        Layer3.biases -= lr * delta3

        delta2 = Layer.backpropagation(a1, z2, Layer2, Layer3, delta3, lr)
        delta1 = Layer.backpropagation(x, z1, Layer1, Layer2, delta2, lr)

    with open("training_progress.txt", "a") as f:
        f.write(f"Epoch {k} - Avg Loss: {loss / 4:.4f} \n")
        print(f"Epoch {k} - Avg Loss: {loss / 4:.4f} \n")
        loss_progress.append(loss)

with open("output.txt", "w") as f:
    for x, y_true in zip(X_train, y_train):
        z1 = Layer1.forward(x)
        a1 = Layer.RelU(z1)
        z2 = Layer2.forward(a1)
        a2 = Layer.RelU(z2)
        z3 = Layer3.forward(a2)
        a3 = Layer.Sigmoid(z3)
        f.write(f"Input: {x} Output: {a3} \n")
        print(f"Input: {x} Output: {a3} \n")

loss_progress = np.array(loss_progress)
epoch_x = np.array(epoch_x)
plt.title("training_loss")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.plot(epoch_x, loss_progress)
plt.show()
