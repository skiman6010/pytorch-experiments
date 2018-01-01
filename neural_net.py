import torch
from torch import autograd, nn, optim
import torch.nn.functional as F

batchsize = 5
inputsize = 4
epochs = 5000
learningrate = 0.001
hiddensize = 10
outputsize = 4

torch.manual_seed(99)
inputdata = autograd.Variable(torch.rand(batchsoze, inputsize))
target = autograd.Variable((torch.rand(batchsize) * outputsize).long())

class net(nn.Module):
    def __init__(self,inputsize, hiddensize, outputsize):
        super(net, self).__init__()
        self.h1 = nn.Linear(inputsize, hiddensize)
        self.h2 = nn.Linear(hiddensize, outputsize)

    def forward(self, x):
        x = self.h1(x)
        x = F.tanh(x)
        x = self.h2(x)
        x = F.log_softmax(x)
        return x

model = net(inputSize = inputsize, hiddenSize = hiddensize, outputSize = outputsize)
optim = optim.Adam(params=model.parameters(), lr = learningrate) #iterator

for epoch in range(epochs):
    out = model(inputdata)
    _,predictdata = out.max(1)
    loss = F.nll_loss(out, target)

    _,predictdata = out.max(1)

    print('predicted data',str(predictdata.view(1,-1)).split('\n')[1])
    print('target data',str(target.view(1,-1)).split('\n')[1])
    print('loss',loss.data[0])

    model.zero_grad()
    loss.backward()
    optim.step()
    if loss.data[0] < 0.1:
        print('Trained successfully.\n')
        break

print('predicted data',str(predictdata.view(1,-1)).split('\n')[1])
print('target data',str(target.view(1,-1)).split('\n')[1])
print('loss',loss.data[0])
