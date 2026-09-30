import unittest
import torch
from methods.gin.src.model import GINCandidate
from methods.gin.src.protocol import masked_cross_entropy, select_validation_threshold, test_metrics

class GINContractTest(unittest.TestCase):
    def test_model_returns_logits_and_backpropagates(self):
        m=GINCandidate(4,64,2,dropout=0.0)
        x=torch.randn(3,4); edge=torch.tensor([[0,1,1,2],[1,0,2,1]])
        y=m(x,edge); self.assertEqual(tuple(y.shape),(3,2)); y.square().mean().backward()
    def test_mask_protocol(self):
        logits=torch.tensor([[2.,0.],[0.,2.],[2.,0.],[0.,2.]],requires_grad=True); labels=torch.tensor([0,1,0,1]); tr=torch.tensor([1,0,0,0],dtype=torch.bool); va=torch.tensor([0,1,0,0],dtype=torch.bool); te=torch.tensor([0,0,1,1],dtype=torch.bool)
        loss=masked_cross_entropy(logits,labels,tr,None); loss.backward(); self.assertEqual(float(logits.grad[~tr].abs().sum()),0.0)
        p=torch.softmax(logits.detach(),1)[:,1]; th,_=select_validation_threshold(labels,p,va); self.assertEqual(test_metrics(labels,p,te,th)['count'],2)
if __name__=='__main__': unittest.main()
