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
train_set = datasets.Spiral()
#plt.scatter(x[:,0], x[:,1], c=t)
#plt.show()
model = MLP((10, 10, 3))
optimizer = optimizers.MomentumSGD(lr, momentum=0.8).setup(model)

data_size = len(train_set)
max_iter = math.ceil(data_size / batch_size)

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

for epoch in range(max_epoch):
    index = np.random.permutation(data_size)
    sum_loss = 0

    for i in range(max_iter):
        # Mini batch
        batch_index = index[i * batch_size:(i + 1) * batch_size]
        batch = [train_set[i] for i in batch_index]
        batch_x = np.array([example[0] for example in batch])
        batch_t = np.array([example[1] for example in batch])

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
