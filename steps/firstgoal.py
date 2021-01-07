import numpy as np

class Variable:
    def __init__(self, data):
        if data is not None:
            if not isinstance(data, np.ndarray):
                raise TypeError(f"The type {type(data)} is not supported.")
        
        self.data = data
        self.grad = None
        self.creator = None
    
    def set_creator(self, func):
        self.creator = func
    
    def cleargrad(self):
        self.grad = None
    
    def backward(self):
        if self.grad is None:
            self.grad = np.ones_like(self.data)
        
        funcs = [self.creator]
        while funcs:
            f = funcs.pop() # 함수를 가져온다
            gys = [output.grad for output in f.outputs]
            gxs = f.backward(*gys)
            if not isinstance(gxs, tuple):
                gxs = (gxs, )
            
            for x, gx in zip(f.inputs, gxs):
                if x.grad is None:
                    x.grad = gx
                else:
                    x.grad = x.grad + gx

                if x.creator is not None:
                    funcs.append(x.creator) # 하나 앞의 함수를 리스트에 추가

class Function:
    def __call__(self, *inputs):
        xs = [x.data for x in inputs]
        ys = self.forward(*xs) # 구체적인 계산은 forward 메서드에서 수행
        if not isinstance(ys, tuple): # 튜플이 아닌 경우 추가 지원
            ys = (ys, )
        outputs = [Variable(as_array(y)) for y in ys]

        for output in outputs:
            output.set_creator(self) # 출력 변수에 창조자를 설정
        self.inputs = inputs # 입력 변수를 기억(보관)
        self.outputs = outputs # 출력도 저장

        return outputs if len(outputs) > 1 else outputs[0]
    
    def forward(self, xs):
        raise NotImplementedError()

    def backward(self, gys):
        raise NotImplementedError()

class Square(Function):
    def forward(self, x):
        return x ** 2

    def backward(self, gy):
        x = self.inputs[0].data
        gx = 2 * x * gy
        return gx

class Exp(Function):
    def forward(self, x):
        return np.exp(x)
    
    def backward(self, gy):
        x = self.inputs[0].data
        gx = np.exp(x) * gy
        return gx

class Add(Function):
    def forward(self, x0, x1):
        y = x0 + x1
        return y
    
    def backward(self, gy):
        return gy, gy

def numerical_diff(f, x, eps=1e-4):
    x0 = Variable(x.data - eps)
    x1 = Variable(x.data + eps)
    y0 = f(x0)
    y1 = f(x1)
    return (y1.data - y0.data) / (2 * eps)

def square(x):
    return Square()(x) # 한 줄로 작성

def exp(x):
    return Exp()(x)

def add(x0, x1):
    return Add()(x0, x1)

def as_array(x):
    if np.isscalar(x):
        return np.array(x)
    return x

x = Variable(np.array(3.0))
y = add(x, x)
print('y', y.data)

y.backward()
print('x.grad', x.grad)