import torch
from torch.autograd import Variable

class MyReLU(torch.autograd.Function):
  
    def forward(self, input):
        self.save_for_backward(input)
        return input.clamp(min=0)
    
    def backward(self, grad_output):
        input = self.saved_tensors
        grad_input = grad_output.clone()
        grad_input[input < 0] = 0
        return grad_input
    
dtype = torch.FloatTensor

# N is batch size; D_in is input dimension;
# H is hidden dimension; D_out is output dimension.
N, D_in, H, D_out = 64, 1000, 100, 10

# Random Tensors for input and output, wrapped in Variables.
x = Variable(torch.randn(N, D_in).type(dtype), requires_grad = False)
y = Variable(torch.randn(N, D_out).type(dtype), requires_grad = False)

# Random Tensors for weights, wrapped in Variables.
w1 = Variable(torch.randn(D_in, H).type(dtype), requires_grad=True)
w2 = Variable(torch.randn(H, D_out).type(dtype), requires_grad=True)

learningrate = 1e-6

for t in range(500):
    y_predict = x.mm(w1).clamp(min=0).mm(w2)
    loss = (y_predict - y).pow(2).sum()
    print(t, loss.data[0])
    
    loss.backward()
    
    w1.data -= learningrate * w1.grad.data
    w2.data -= learningrate * w2.grad.data

    w1.grad.data.zero_()
    w2.grad.data.zero_()
