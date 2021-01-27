import torch.nn as nn
import torch
import math
class EasyModel(nn.Module):
    def __init__(self, class_num):
        self.name = 'EasyModel'
        super(EasyModel, self).__init__()
        self.block1 = nn.Sequential(nn.Conv2d(3, 32, kernel_size=3, stride=1, padding=1),
                                   nn.BatchNorm2d(32),
                                   nn.ReLU(),)
        self.block2 = nn.Sequential(nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
                                   nn.BatchNorm2d(64),
                                   nn.ReLU(),)                        
        self.block3 = nn.Sequential(nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
                                   nn.BatchNorm2d(128),
                                   nn.ReLU(),)                          
        self.block4 = nn.Sequential(nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1),
                                   nn.BatchNorm2d(256),
                                   nn.ReLU(),)   
        self.avg_pool = nn.AdaptiveMaxPool2d((1))  
        self.dropout = nn.Dropout(0.5)
        self.fc1 = nn.Linear(256,128)   
        self.fc2 = nn.Linear(128,64)         
        self.fc3 = nn.Linear(64,class_num)         
                                   
                                
        #self._init_weight()
        
    def forward(self, inputs ): 
        bs = inputs.size(0)
        
        x = self.block1(inputs)
        #print(x.size())
        x = self.block2(x)
        #print(x.size())
        x = self.block3(x)
        #print(x.size())
        x = self.block4(x)
        #print(x.size())
        x = self.avg_pool(x)
        #print(x.size())
        x = x.view(bs, -1)
        #x = self.dropout(x)
        x = self.fc1(x)
        x = self.fc2(x)
        outputs = self.fc3(x)
        
        return outputs    
    def _init_weight(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))            
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            