# -*- coding: utf-8 -*-
"""
Created on Fri Oct 29 10:05:46 2021

@author: Saif
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

class Siamese3DNet(nn.Module):
    
    def __init__(self,input_channels,):
        super(Siamese3DNet, self).__init__()
        self.conv1 = nn.Conv3d(input_channels, out_channels=64, kernel_size=7)
        self.conv2 = nn.Conv3d(64, 128, 7)
        self.conv3 = nn.Conv3d(128, 128, 4)
        self.conv4 = nn.Conv3d(128, 256, 4)

    
    def before_forward(self,x):
        
        out_block1 = F.relu(F.max_pool3d(self.conv1(x),2))
        out_block2 = F.relu(F.max_pool3d(self.conv2(out_block1),2))
        out_block3 = F.relu(F.max_pool3d(self.conv3(out_block2),2))
        out_block4 = F.relu(F.max_pool3d(self.conv4(out_block3),2))
        
        return out_block1,out_block2,out_block3,out_block4
    
    def forward (self,x1,x2):
        
        out_block1_1,out_block2_1,out_block3_1,out_block4_1 = self.before_forward(x1)
        out_block1_2,out_block2_2,out_block3_2,out_block4_2 = self.before_forward(x2) 
        
        out_block1_1 = out_block1_1.flatten(start_dim=1)
        out_block2_1 = out_block2_1.flatten(start_dim=1)
        out_block3_1 = out_block3_1.flatten(start_dim=1)
        out_block4_1 = out_block4_1.flatten(start_dim=1)
        
        Feature_1 = torch.cat((out_block1_1,out_block2_1,out_block3_1,out_block4_1),1) #output (#Batches,flat)
        
        
        out_block1_2 = out_block1_2.flatten(start_dim=1)
        out_block2_2 = out_block2_2.flatten(start_dim=1)
        out_block3_2 = out_block3_2.flatten(start_dim=1)
        out_block4_2 = out_block4_2.flatten(start_dim=1)
        
        Feature_2 = torch.cat((out_block1_2,out_block2_2,out_block3_2,out_block4_2),1) # #output (#Batches,flat)
        
        pdist = nn.PairwiseDistance(p=2,keepdim=True)
        
        return pdist(Feature_1, Feature_2)
        
        
        
        
        
        
        
    