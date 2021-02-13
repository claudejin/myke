import math
import numpy as np
import myke
from myke.datasets import Spiral
from myke import DataLoader
from myke import optimizers
from myke.models import MLP
import myke.functions as F
import matplotlib.pyplot as plt

# Hyperparameter settings
max_epoch = 10
batch_size = 10
lr = 0.4

# Data
train_set = Spiral(train=True)
test_set = Spiral(train=False)
train_loader = DataLoader(train_set, batch_size)
test_loader = DataLoader(test_set, batch_size, shuffle=False)

model = MLP((10, 10, 3))
optimizer = optimizers.MomentumSGD(lr, momentum=0.8).setup(model)

"""
def draw_model():
    # Plot boundary area the model predict
    h = 0.01
    x_min, x_max = -1, +1
    y_min, y_max = -1, +1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))
    X = np.c_[xx.ravel(), yy.ravel()]

    with myke.no_grad():
        score = model(X)
    predict_cls = np.argmax(score.data, axis=1)
    Z = predict_cls.reshape(xx.shape)
    plt.contourf(xx, yy, Z)

    # Plot data points of the dataset
    N, CLS_NUM = 100, 3
    markers = ['o', 'x', '^']
    colors = ['orange', 'blue', 'green']
    for i in range(data_size):
        x, t = train_set[i]
        c = t
        plt.scatter(x[0], x[1], s=40,  marker=markers[c], c=colors[c])
    plt.draw()
    plt.pause(0.01)
    plt.clf()
"""

for epoch in range(max_epoch):
    sum_loss, sum_acc = 0, 0

    for x, t in train_loader: # Batch iter
        y = model(x)
        loss = F.softmax_cross_entropy_simple(y, t)
        acc = F.accuracy(y, t)

        model.cleargrads()
        loss.backward()
        optimizer.update()

        sum_loss += float(loss.data) * len(t)
        sum_acc += float(acc.data) * len(t)

    print(f"epoch {epoch+1}")
    print(f"train loss {sum_loss / len(train_set):.4f}, accuracy {sum_acc / len(train_set):.4f}")
    
    sum_loss, sum_acc = 0, 0
    with myke.no_grad():
        for x, t in test_loader:
            y = model(x)
            loss = F.softmax_cross_entropy_simple(y, t)
            acc = F.accuracy(y, t)

            sum_loss += float(loss.data) * len(t)
            sum_acc += float(acc.data) * len(t)
        
    print(f"test loss {sum_loss / len(test_set):.4f}, accuracy {sum_acc / len(test_set):.4f}")

    # if epoch % 250 == 0:
    #    draw_model()
    