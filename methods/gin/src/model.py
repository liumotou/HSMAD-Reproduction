from __future__ import annotations
import torch
from torch import nn
from torch_geometric.nn import GINConv
class GINCandidate(nn.Module):
    def __init__(self,input_dim:int,hidden_dim:int,output_dim:int,dropout:float=0.0):
        super().__init__()
        self.conv1=GINConv(nn.Sequential(nn.Linear(input_dim,hidden_dim),nn.ReLU(),nn.Linear(hidden_dim,hidden_dim)),eps=0.,train_eps=False)
        self.conv2=GINConv(nn.Sequential(nn.Linear(hidden_dim,hidden_dim),nn.ReLU(),nn.Linear(hidden_dim,hidden_dim)),eps=0.,train_eps=False)
        self.dropout=nn.Dropout(dropout); self.classifier=nn.Linear(hidden_dim,output_dim)
    def forward(self,x:torch.Tensor,edge_index:torch.Tensor)->torch.Tensor:
        x=torch.relu(self.conv1(x,edge_index)); x=self.dropout(x); x=torch.relu(self.conv2(x,edge_index)); return self.classifier(x)
