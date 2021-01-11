import numpy as np
from myke import Variable

def rosenbrock(x0, x1):
    y = 100 * (x1 - x0 ** 2) ** 2 + (1 - x0) ** 2
    return y

def gradient_descent():
    x0 = Variable(np.array(0.0))
    x1 = Variable(np.array(2.0))
    lr = 0.001 # Learning rate
    iters = 1000 # iterations

    for i in range(iters):
        print(i, x0, x1)
        
        y = rosenbrock(x0, x1)

        x0.cleargrad()
        x1.cleargrad()
        y.backward()
        
        x0.data -= lr * x0.grad
        x1.data -= lr * x1.grad

def f(x):
    y = x ** 4 - 2 * x ** 2
    return y

def gx2(x):
    return 12 * x ** 2 - 4

def newton_method():
    x = Variable(np.array(2.0))
    iters = 10

    for i in range(iters):
        print(i, x)

        y = f(x)
        x.cleargrad()
        y.backward()

        x.data -= x.grad / gx2(x.data)

def secondorder_method():
    x = Variable(np.array(2.0))
    iters = 10

    for i in range(iters):
        print(i, x)

        y = f(x)
        x.cleargrad()
        y.backward(create_graph=True)

        gx = x.grad
        x.cleargrad()
        gx.backward(create_graph=True)
        
        gx2 = x.grad

        x.data -= gx.data / gx2.data

secondorder_method()