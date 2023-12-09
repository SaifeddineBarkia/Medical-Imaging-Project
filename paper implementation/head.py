# -*- coding: utf-8 -*-
"""
Created on Fri Oct 29 12:24:47 2021

@author: Saif
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

class head(nn.Module):
    def __init__(self,backbone):
        super(head,self).__init__()
        self.backbone = backbone
        
    def forward(self,x1,x2):
        
        out_block1_1,out_block2_1,out_block3_1,out_block4_1 = self.backbone(x1)
        out_block1_2,out_block2_2,out_block3_2,out_block4_2 = self.backbone(x2) 
        
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
        
        pdist = nn.PairwiseDistance(p=1,keepdim=True) #keep_dim = True to keep the number of batches
        
        return pdist(Feature_1,Feature_2)
        