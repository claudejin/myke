import numpy as np
from myke import Variable
from myke.utils import plot_dot_graph
import myke.functions as F

x = Variable(np.array(1.0), 'x')
y = F.tanh(x)
y.name = 'y'
y.backward(create_graph=True)

iters = 0

for i in range(iters):
    gx = x.grad
    x.cleargrad()
    gx.backward(create_graph=True)

gx = x.grad
gx.name = 'gx' + str(iters+1)
plot_dot_graph(gx, verbose=False, to_file='tanh.png')

# ex2
x = Variable(np.array(2.0))
y = x ** 2
y.backward(create_graph=True)
gx = x.grad
x.cleargrad()
z = gx ** 3 + y
z.backward()
print(x.grad)
