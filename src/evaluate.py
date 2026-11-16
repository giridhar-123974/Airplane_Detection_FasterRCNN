"""Evaluate detections using class-wise AP at IoU 0.5."""
from __future__ import annotations
import argparse
from pathlib import Path
import numpy as np, pandas as pd, torch
from torch.utils.data import DataLoader
from .dataset import AircraftDataset, create_splits, collate_fn
from .model import build_model
from .transforms import get_transforms
from .utils import get_device, setup_logging

def evaluate_ap(model, loader, device, class_names, iou_threshold=.5):
    model.eval(); detections={c:[] for c in range(1,len(class_names)+1)}; gt_counts={c:0 for c in detections}
    with torch.no_grad():
        for images, targets in loader:
            outputs=model([image.to(device) for image in images])
            for out,target in zip(outputs,targets):
                tid=int(target['image_id'].item())
                for label in target['labels'].tolist(): gt_counts[label]+=1
                for box,label,score in zip(out['boxes'].cpu(),out['labels'].cpu(),out['scores'].cpu()):
                    detections[int(label)].append((float(score),tid,box.numpy()))
    aps={}
    for cls in detections:
        gt={}; matched={}
        # Targets are retained by image ID using a second loader pass for clarity.
        for _,targets in loader:
            for target in targets:
                tid=int(target['image_id'].item()); mask=target['labels']==cls
                gt[tid]=target['boxes'][mask].numpy(); matched[tid]=np.zeros(mask.sum(),dtype=bool)
        tp=[]; fp=[]
        for score,tid,box in sorted(detections[cls],key=lambda x:x[0],reverse=True):
            boxes=gt.get(tid,np.empty((0,4)))
            if len(boxes):
                xx1=np.maximum(box[0],boxes[:,0]); yy1=np.maximum(box[1],boxes[:,1])
                xx2=np.minimum(box[2],boxes[:,2]); yy2=np.minimum(box[3],boxes[:,3])
                inter=np.maximum(0,xx2-xx1)*np.maximum(0,yy2-yy1)
                union=(box[2]-box[0])*(box[3]-box[1])+(boxes[:,2]-boxes[:,0])*(boxes[:,3]-boxes[:,1])-inter
                ious=inter/np.maximum(union,1e-9); best=int(ious.argmax())
                hit=ious[best]>=iou_threshold and not matched[tid][best]
            else: hit=False
            tp.append(float(hit)); fp.append(float(not hit))
            if hit: matched[tid][best]=True
        if gt_counts[cls]==0: aps[class_names[cls-1]]=float('nan'); continue
        tp=np.cumsum(tp); fp=np.cumsum(fp); recall=tp/gt_counts[cls]; precision=tp/np.maximum(tp+fp,1e-9)
        recall=np.r_[0,recall,1]; precision=np.r_[0,precision,0]
        for i in range(len(precision)-2,-1,-1): precision[i]=max(precision[i],precision[i+1])
        idx=np.where(recall[1:]!=recall[:-1])[0]
        aps[class_names[cls-1]]=float(np.sum((recall[idx+1]-recall[idx])*precision[idx+1]))
    aps['mAP@0.5']=float(np.nanmean(list(aps.values())))
    return aps

def main():
    p=argparse.ArgumentParser(description=__doc__); p.add_argument('--dataset',default='dataset'); p.add_argument('--checkpoint',default='checkpoints/best.pth'); p.add_argument('--batch-size',type=int,default=2); p.add_argument('--workers',type=int,default=2); a=p.parse_args()
    setup_logging(); _,val_csv=create_splits(a.dataset)
    checkpoint=torch.load(a.checkpoint,map_location='cpu',weights_only=False); names=checkpoint['class_names']
    ds=AircraftDataset(a.dataset,pd.read_csv(val_csv),get_transforms(False),class_names=names); loader=DataLoader(ds,batch_size=a.batch_size,shuffle=False,num_workers=a.workers,collate_fn=collate_fn)
    device=get_device(); model=build_model(len(names),False).to(device); model.load_state_dict(checkpoint['model_state_dict'])
    result=evaluate_ap(model,loader,device,names); print('\n'.join(f'{k}: {v:.4f}' for k,v in result.items()))

if __name__=='__main__': main()
