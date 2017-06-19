import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

class  CNN_Text(nn.Module):
    
    def __init__(self, args):
        super(CNN_Text,self).__init__()
        self.args = args
        
        V = args.embed_num#37
        D = args.embed_dim#128
        C = args.class_num#
        Ci = 1
        Co = args.kernel_num#100
        Ks = args.kernel_sizes#3,4,5

        self.embed = nn.Embedding(V, D)
        self.convs1 = [nn.Conv2d(Ci, Co, (K, D)) for K in Ks]
        '''
        self.conv13 = nn.Conv2d(Ci, Co, (3, D))
        self.conv14 = nn.Conv2d(Ci, Co, (4, D))
        self.conv15 = nn.Conv2d(Ci, Co, (5, D))
        '''
        self.dropout = nn.Dropout(args.dropout)
        self.fc1 = nn.Linear(len(Ks)*Co, C)

    def conv_and_pool(self, x, conv):
        x = F.relu(conv(x)).squeeze(3) #(N,Co,W)
        x = F.max_pool1d(x, x.size(2)).squeeze(2)
        return x


    def forward(self, x):
        #x[1803 x 37]
        x = self.embed(x) # (N,W,D)
        #print(x)词向量[1803 x 37 x 128]
        if self.args.static:
            x = Variable(x)
        #print(x)
        x = x.unsqueeze(1) # (N,Ci,W,D)
        #print(x),#增加了一个维度1803  1 x 37 x 128

        #x = F.relu(self.convs1[0](x)).squeeze(3)
        #print(x)#会得到[1803 x 100 x 35 x 1][1803 x 100 x 34 x 1][1803 x 100 x 33 x 1]

        x = [F.relu(conv(x)).squeeze(3) for conv in self.convs1] #[(N,Co,W), ...]*len(Ks)
        #print(x)#[1803 x 100 x 35][1803 x 100 x 34][1803 x 100 x 33]

        # print(x[0].size(2))
        # x = F.max_pool1d(x[0],x[0].size(2)).squeeze(2)
        # print(x)

        x = [F.max_pool1d(i, i.size(2)).squeeze(2) for i in x] #[(N,Co), ...]*len(Ks)
        #[1803 x 100][1803 x 100][1803 x 100]
        x = torch.cat(x, 1)#[1803 x 300]
        #print(x)

        '''
        x1 = self.conv_and_pool(x,self.conv13) #(N,Co)
        x2 = self.conv_and_pool(x,self.conv14) #(N,Co)
        x3 = self.conv_and_pool(x,self.conv15) #(N,Co)
        x = torch.cat((x1, x2, x3), 1) # (N,len(Ks)*Co)
        '''
        x = self.dropout(x) # (N,len(Ks)*Co)[1803 x 300]
        logit = self.fc1(x) # (N,C)[1803 x 2]
        return logit