import torch
import torch.nn.functional as F
from torch.optim import lr_scheduler
from torch import nn, optim
        
class BasicBlock(nn.Module):
    def __init__(self, input_size, hidden_size):
        super(BasicBlock, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.layer1 = nn.Sequential(nn.Linear(self.input_size, self.hidden_size),
                                    nn.BatchNorm1d(self.hidden_size),
                                    nn.ReLU())
        self.layer2 = nn.Sequential(nn.Linear(self.hidden_size, self.hidden_size),
                                    nn.BatchNorm1d(self.hidden_size),
                                    nn.ReLU())
        self.layer3 = nn.Sequential(nn.Linear(self.hidden_size, self.hidden_size),
                                    nn.BatchNorm1d(self.hidden_size),
                                    nn.ReLU())                            
        self.layer4 = nn.Sequential(nn.Linear(self.hidden_size, self.input_size),
                                    nn.BatchNorm1d(self.input_size))
        self.relu = nn.ReLU()
    
    def forward(self, x):
        out = self.layer1(x)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = out + x
        out = self.relu(out)
        return out
      
class BasicBlockX(nn.Module):
    def __init__(self, input_size, hidden_size):
        super(BasicBlockX, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.bottleneck1 = BasicBlock(self.input_size, self.hidden_size)
        self.bottleneck2 = BasicBlock(self.input_size, self.hidden_size)
        # self.bottleneck3 = BasicBlock(self.input_size, self.hidden_size)
        self.relu = nn.ReLU()
    
    def forward(self, x):
        out1 = self.bottleneck1(x)
        out2 = self.bottleneck2(x)
        # out3 = self.bottleneck3(x)
        out = out1 + out2 + x
        out = self.relu(out)
        return out

class MyResnet(nn.Module):
    def __init__(self, input_size):
        super(MyResnet, self).__init__()
        self.layer_size = 128
        self.hidden_size = self.layer_size * 5
        self.inputblock = nn.Sequential(nn.Linear(input_size, self.layer_size),
                                nn.BatchNorm1d(self.layer_size),
                                nn.ReLU())
        self.basicblock1 = BasicBlock(self.layer_size, self.hidden_size)
        self.basicblock2 = BasicBlock(self.layer_size, self.hidden_size)
        self.basicblock3 = BasicBlock(self.layer_size, self.hidden_size)
        self.basicblock4 = BasicBlock(self.layer_size, self.hidden_size)
        self.basicblock5 = BasicBlock(self.layer_size, self.hidden_size)
        self.basicblock6 = BasicBlock(self.layer_size, self.hidden_size)
        self.dropout = nn.Dropout(0.1)
        self.attention1 = nn.Sequential(nn.Linear(self.layer_size, self.layer_size//16),
                                nn.ReLU(),
                                nn.Linear(self.layer_size//16, self.layer_size),
                                nn.Sigmoid())
        self.attention2 = nn.Sequential(nn.Linear(self.layer_size, self.layer_size//16),
                                nn.ReLU(),
                                nn.Linear(self.layer_size//16, self.layer_size),
                                nn.Sigmoid())
        self.outputblock = nn.Linear(self.layer_size, 1)

        
    def forward(self, x):
        out = self.inputblock(x)
        out = self.basicblock1(out)
        out = self.basicblock2(out)
        atten1 = self.attention1(out)
        out = torch.mul(out, atten1)
        
        out = self.basicblock3(out)
        out = self.basicblock4(out)
        atten2 = self.attention2(out)
        out = self.dropout(out)
        out = torch.mul(out, atten2)
        out = self.outputblock(out)
        return out
        
def buildModel(model, model_path):
    model.load_state_dict(torch.load(model_path))
    return model
        