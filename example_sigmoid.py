import numpy as np
from myke import Variable
from myke import optimizers
import myke.functions as F
import myke.layers as L
import myke.models as M
import matplotlib.pyplot as plt

# 데이터셋
np.random.seed(0)
x = np.random.rand(100, 1)
y = np.sin(2 * np.pi * x) + np.random.rand(100, 1)

lr = 0.2
iters = 10000
hidden_size = 10

model = M.MLP((hidden_size, 1))
optimizer = optimizers.MomentumSGD(lr).setup(model)

# 신경망 학습
for i in range(iters):
    y_pred = model(x)
    loss = F.mean_squared_error(y, y_pred)

    model.cleargrads()
    loss.backward()

    optimizer.update()
    
    if i % 1000 == 0: # 1000회마다 출력
        print(loss)

plt.scatter(x, y)

x = np.sort(x.reshape(100,)).reshape(100, 1)
plt.plot(x, model(x).data, color='r')
plt.show()
