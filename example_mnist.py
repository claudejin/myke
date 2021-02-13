import numpy as np
import myke
from myke import datasets
from myke import DataLoader
from myke.models import MLP
from myke import optimizers
from myke import functions as F

max_epoch = 5
batch_size = 100

train_set = datasets.MNIST(train=True)
test_set = datasets.MNIST(train=False)
train_loader = DataLoader(train_set, batch_size)
test_loader = DataLoader(test_set, batch_size, shuffle=False)

model = MLP((1000, 10), activation=F.relu)
optimizer = optimizers.MomentumSGD().setup(model)

for epoch in range(max_epoch):
    sum_loss, sum_acc = 0, 0

    for x, t in train_loader:
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
