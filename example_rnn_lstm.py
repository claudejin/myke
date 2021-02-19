import numpy as np
import myke
from myke.datasets import SinCurve
from myke.dataloaders import SeqDataLoader
from myke.models import SimpleRNN
from myke.optimizers import Adam
import myke.functions as F
import matplotlib.pyplot as plt

max_epoch = 100
batch_size = 30
hidden_size = 100
bptt_length = 30

train_set = SinCurve(train=True)
dataloader = SeqDataLoader(train_set, batch_size=batch_size)
seqlen = len(train_set)

model = SimpleRNN(hidden_size, 1, unit='lstm')
optimizer = Adam().setup(model)

for epoch in range(max_epoch):
    model.reset_state()
    loss, count = 0, 0

    for x, t in dataloader:
        y = model(x)
        loss += F.mean_squared_error(y, t)
        count += 1

        if count % bptt_length == 0 or count == seqlen:
            model.cleargrads()
            loss.backward()
            loss.unchain_backward()
            optimizer.update()
    
    avg_loss = float(loss.data) / count
    print(f"| epoch {epoch+1:03d} | loss {avg_loss:.6f}")

xs = np.cos(np.linspace(0, 4 * np.pi, 1000))
model.reset_state()
pred_list = []

with myke.test_mode() and myke.no_grad():
    for x in xs:
        x = np.array(x).reshape(1, 1)
        y = model(x)
        pred_list.append(float(y.data))

plt.plot(np.arange(len(xs)), xs, label='y=cos(x)')
plt.plot(np.arange(len(xs)), pred_list, label='predict')
plt.xlabel('x')
plt.ylabel('y')
plt.legend()
plt.show()
