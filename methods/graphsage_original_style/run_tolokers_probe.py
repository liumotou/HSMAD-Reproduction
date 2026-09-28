import json, time, hashlib, csv
from pathlib import Path
import dgl, torch
import torch.nn as nn
import torch.nn.functional as F
from dgl.nn import SAGEConv
from sklearn.metrics import f1_score, roc_auc_score, average_precision_score

ROOT=Path(__file__).resolve().parents[2]
OUT=ROOT/'results/experiments/graphsage/tolokers/graphsage_original_style_h64_sampling_probe/diagnostic/seed_0'
def sha(p):
 h=hashlib.sha256();h.update(Path(p).read_bytes());return h.hexdigest()
class Net(nn.Module):
 def __init__(self,d): super().__init__();self.a=SAGEConv(d,64,'pool',activation=F.relu);self.b=SAGEConv(64,64,'pool',activation=F.relu);self.c=nn.Linear(64,2)
 def forward(self,g,x): return self.c(self.b(g,self.a(g,x)))
 def block(self,bs,x): return self.c(self.b(bs[1],self.a(bs[0],x)))
def met(y,p,t):
 q=p>=t;return {'f1_macro':float(f1_score(y,q,average='macro',zero_division=0)),'auroc':float(roc_auc_score(y,p)),'auprc':float(average_precision_score(y,p)),'predicted_anomaly_count':int(q.sum())}
def main():
 if OUT.exists(): raise FileExistsError(OUT)
 OUT.mkdir(parents=True); cfg={'protocol':'graphsage_original_style_h64_sampling_probe','dataset':'tolokers','seed':0,'fanout':[25,10],'batch_size':512,'h_feats':64,'layers':2,'aggregation':'pool','max_epoch':200,'patience':50,'train_sampling':'DGL NeighborSampler([25,10])','evaluation':'full_graph','positioning':'diagnostic; not author-exact'}; (OUT/'config.json').write_text(json.dumps(cfg,indent=2))
 torch.manual_seed(0);dgl.seed(0);raw=dgl.load_graphs(str(ROOT/'datasets/tolokers'))[0][0];g=dgl.add_self_loop(dgl.remove_self_loop(dgl.to_bidirected(raw)));x=raw.ndata['feature'];y=raw.ndata['label'].long();m={k:raw.ndata[k].bool() for k in ['train_mask','val_mask','test_mask']}; ids=torch.nonzero(m['train_mask'],as_tuple=False).squeeze();w=torch.tensor([1.,float((y[m['train_mask']]==0).sum()/y[m['train_mask']].sum())]); pre={'fanout':[25,10],'batch_size':512,'graph':{'nodes':g.num_nodes(),'edges':g.num_edges()},'mask_counts':{k:int(v.sum()) for k,v in m.items()},'class_weight':w.tolist(),'model':'two pool SAGEConv h=64 + Linear logits','loss_mask':'train_mask only','selection_mask':'val_mask only','test_mask':'test_mask only','script_sha256':sha(__file__)};(OUT/'preflight.json').write_text(json.dumps(pre,indent=2))
 dev='cuda';g=g.to(dev);x=x.to(dev);y=y.to(dev);m={k:v.to(dev) for k,v in m.items()};model=Net(x.shape[1]).to(dev);opt=torch.optim.Adam(model.parameters(),lr=.01,weight_decay=0);sam=dgl.dataloading.NeighborSampler([25,10]);loader=dgl.dataloading.DataLoader(g,ids.to(dev),sam,batch_size=512,shuffle=True,drop_last=False,num_workers=0,device=dev);hist=[];best=-1;pc=0;ck=OUT/'checkpoint.pt';torch.cuda.reset_peak_memory_stats()
 for e in range(1,201):
  model.train();losses=[]
  for inp,out,bs in loader:
   z=model.block(bs,x[inp]);loss=F.cross_entropy(z,y[out],weight=w.to(dev));opt.zero_grad();loss.backward();opt.step();losses.append(loss.item())
  model.eval();p=torch.softmax(model(g,x),1)[:,1].detach().cpu().numpy();vy=y[m['val_mask']].cpu().numpy();vp=p[m['val_mask'].cpu().numpy()];ts=[i/20 for i in range(1,20)];t=max(ts,key=lambda z:f1_score(vy,vp>=z,average='macro',zero_division=0));v=met(vy,vp,t);imp=v['auprc']>best;best=v['auprc'] if imp else best;pc=0 if imp else pc+1
  if imp: torch.save({'epoch':e,'model':model.state_dict()},ck)
  hist.append({'epoch':e,'train_loss':sum(losses)/len(losses),'validation':v,'threshold':t,'patience':pc})
  if pc>50: break
 saved=torch.load(ck);model.load_state_dict(saved['model']);p=torch.softmax(model(g,x),1)[:,1].detach().cpu().numpy();vi=m['val_mask'].cpu().numpy();ti=m['test_mask'].cpu().numpy();vy=y[m['val_mask']].cpu().numpy();vp=p[vi];t=max([i/20 for i in range(1,20)],key=lambda z:f1_score(vy,vp>=z,average='macro',zero_division=0));res=met(y[m['test_mask']].cpu().numpy(),p[ti],t);res.update({'best_epoch':saved['epoch'],'threshold':t,'actual_epochs':len(hist),'peak_gpu_mb':torch.cuda.max_memory_allocated()/1024**2});(OUT/'validation_history.json').write_text(json.dumps(hist,indent=2));(OUT/'metrics.json').write_text(json.dumps(res,indent=2));(OUT/'runs.csv').write_text('dataset,seed,status,f1_macro,auroc\ntolokers,0,diagnostic,'+str(res['f1_macro'])+','+str(res['auroc'])+'\n');print(json.dumps(res))
if __name__=='__main__': main()
