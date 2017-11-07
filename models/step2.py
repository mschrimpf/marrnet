
import torch
import torch.nn as nn
from torch.autograd import Variable
from functools import reduce

class LambdaBase(nn.Sequential):
    def __init__(self, fn, *args):
        super(LambdaBase, self).__init__(*args)
        self.lambda_func = fn

    def forward_prepare(self, input):
        output = []
        for module in self._modules.values():
            output.append(module(input))
        return output if output else input

class Lambda(LambdaBase):
    def forward(self, input):
        return self.lambda_func(self.forward_prepare(input))

class LambdaMap(LambdaBase):
    def forward(self, input):
        return list(map(self.lambda_func,self.forward_prepare(input)))

class LambdaReduce(LambdaBase):
    def forward(self, input):
        return reduce(self.lambda_func,self.forward_prepare(input))


step2 = nn.Sequential( # Sequential,
	nn.Conv2d(4,96,(11, 11),(4, 4),(0, 0),1,1,bias=False),
	nn.BatchNorm2d(96),
	nn.ReLU(),
	nn.MaxPool2d((3, 3),(2, 2)),
	nn.Conv2d(96,256,(5, 5),(1, 1),(2, 2),1,2,bias=False),
	nn.BatchNorm2d(256),
	nn.ReLU(),
	nn.MaxPool2d((3, 3),(2, 2)),
	nn.Conv2d(256,384,(3, 3),(1, 1),(1, 1),1,1,bias=False),
	nn.BatchNorm2d(384),
	nn.ReLU(),
	nn.Conv2d(384,384,(3, 3),(1, 1),(1, 1),1,2,bias=False),
	nn.BatchNorm2d(384),
	nn.ReLU(),
	nn.Conv2d(384,256,(3, 3),(1, 1),(1, 1),1,2,bias=False),
	nn.BatchNorm2d(256),
	nn.ReLU(),
	nn.MaxPool2d((3, 3),(2, 2)),
	Lambda(lambda x: x.view(x.size(0),-1)), # Reshape,
	nn.Sequential(Lambda(lambda x: x.view(1,-1) if 1==len(x.size()) else x ),nn.Linear(9216,4096)), # Linear,
	nn.ReLU(),
	nn.Dropout(0.5),
	nn.Sequential(Lambda(lambda x: x.view(1,-1) if 1==len(x.size()) else x ),nn.Linear(4096,4096)), # Linear,
	nn.ReLU(),
	nn.Dropout(0.5),
	nn.Sequential(Lambda(lambda x: x.view(1,-1) if 1==len(x.size()) else x ),nn.Linear(4096,200)), # Linear,
	Lambda(lambda x: x.view(x.size(0),-1,1,1,1)), # Reshape,
	nn.ConvTranspose3d(200,512,(4, 4, 4),(1, 1, 1),(0, 0, 0),(0, 0, 0)),
	nn.BatchNorm3d(512,1e-05,0.1,True),#BatchNorm3d,
	nn.ReLU(),
	nn.ConvTranspose3d(512,256,(4, 4, 4),(2, 2, 2),(1, 1, 1),(0, 0, 0)),
	nn.BatchNorm3d(256,1e-05,0.1,True),#BatchNorm3d,
	nn.ReLU(),
	nn.ConvTranspose3d(256,128,(4, 4, 4),(2, 2, 2),(1, 1, 1),(0, 0, 0)),
	nn.BatchNorm3d(128,1e-05,0.1,True),#BatchNorm3d,
	nn.ReLU(),
	nn.ConvTranspose3d(128,64,(4, 4, 4),(2, 2, 2),(1, 1, 1),(0, 0, 0)),
	nn.BatchNorm3d(64,1e-05,0.1,True),#BatchNorm3d,
	nn.ReLU(),
	nn.ConvTranspose3d(64,32,(4, 4, 4),(2, 2, 2),(1, 1, 1),(0, 0, 0)),
	nn.BatchNorm3d(32,1e-05,0.1,True),#BatchNorm3d,
	nn.ReLU(),
	nn.ConvTranspose3d(32,1,(4, 4, 4),(2, 2, 2),(1, 1, 1),(0, 0, 0)),
	nn.Sigmoid(),
)