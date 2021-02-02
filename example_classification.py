import math
import numpy as np
import myke
from myke import datasets
from myke import optimizers
from myke.models import MLP
import myke.functions as F
import matplotlib.pyplot as plt

# Hyperparameter settings
max_epoch = 3000
batch_size = 60
lr = 0.4

# Data
x, t = datasets.get_spiral(train=True)
#plt.scatter(x[:,0], x[:,1], c=t)
#plt.show()
model = MLP((10, 10, 3))
optimizer = optimizers.MomentumSGD(lr, momentum=0.8).setup(model)

data_size = len(x)
max_iter = math.ceil(data_size / batch_size)

def draw_model():
    # Plot boundary area the model predict
    h = 0.01
    x_min, x_max = x[:, 0].min() - .1, x[:, 0].max() + .1
    y_min, y_max = x[:, 1].min() - .1, x[:, 1].max() + .1
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
    for i in range(len(x)):
        c = t[i]
        plt.scatter(x[i][0], x[i][1], s=40,  marker=markers[c], c=colors[c])
    plt.draw()
    plt.pause(0.01)
    plt.clf()

for epoch in range(max_epoch):
    index = np.random.permutation(data_size)
    sum_loss = 0

    for i in range(max_iter):
        # Mini batch
        batch_index = index[i * batch_size:(i + 1) * batch_size]
        batch_x = x[batch_index]
        batch_t = t[batch_index]

        y = model(batch_x)
        loss = F.softmax_cross_entropy_simple(y, batch_t)
        
        model.cleargrads()
        loss.backward()
        optimizer.update()

        sum_loss += float(loss.data) * len(batch_t)
    
    # Print progress every epoch
    avg_loss = sum_loss / data_size
    print(f"epoch {epoch+1}, loss {avg_loss:.6f}")
    if epoch % 250 == 0:
        draw_model()
