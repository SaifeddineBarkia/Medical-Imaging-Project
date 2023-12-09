# -*- coding: utf-8 -*-
"""
Created on Fri Oct 29 12:22:03 2021

@author: Saif
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class Siamse3Dnetwork(nn.Module):
    def __init__(self,backbone,head):
        self.backbone = backbone
        self.head = head
        
    def forward(self,x1,x2):    
        
        return self.head(self.backbone)(x1,x2)