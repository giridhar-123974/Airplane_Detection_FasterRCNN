"""Train Faster R-CNN on the Airbus aircraft dataset."""
from __future__ import annotations
import argparse, logging, time
from pathlib import Path
import pandas as pd
import torch
from torch.utils.data import DataLoader
from .dataset import AircraftDataset, create_splits, collate_fn
from .engine import train_one_epoch, validation_loss
from .model import build_model
from .transforms import get_transforms
from .utils import get_device, save_checkpoint, seed_everything, setup_logging
from .visualization import plot_history

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--dataset',default='dataset'); parser.add_argument('--epochs',type=int,default=20)
    parser.add_argument('--batch-size',type=int,default=2); parser.add_argument('--lr',type=float,default=.005)
    parser.add_argument('--workers',type=int,default=2); parser.add_argument('--seed',type=int,default=42)
    parser.add_argument('--resume',default=None); parser.add_argument('--no-pretrained',action='store_true')
    parser.add_argument('--output',default='outputs'); parser.add_argument('--checkpoints',default='checkpoints')
    args=parser.parse_args(); setup_logging(); seed_everything(args.seed); log=logging.getLogger(__name__)
    root=Path(args.dataset); train_csv,val_csv=create_splits(root,seed=args.seed)
    from .dataset import inspect_annotations
    _, class_to_id = inspect_annotations(root / 'annotations.csv')
    class_names = list(class_to_id)
    train_ds=AircraftDataset(root,pd.read_csv(train_csv),get_transforms(True),class_names=class_names)
    val_ds=AircraftDataset(root,pd.read_csv(val_csv),get_transforms(False),class_names=class_names)
    train_loader=DataLoader(train_ds,batch_size=args.batch_size,shuffle=True,num_workers=args.workers,
                            collate_fn=collate_fn,pin_memory=torch.cuda.is_available())
    val_loader=DataLoader(val_ds,batch_size=args.batch_size,shuffle=False,num_workers=args.workers,
                          collate_fn=collate_fn,pin_memory=torch.cuda.is_available())
    device=get_device(); model=build_model(len(train_ds.classes),not args.no_pretrained).to(device)
    optimizer=torch.optim.SGD([p for p in model.parameters() if p.requires_grad],lr=args.lr,momentum=.9,weight_decay=.0005)
    scheduler=torch.optim.lr_scheduler.StepLR(optimizer,step_size=5,gamma=.1)
    scaler=torch.amp.GradScaler('cuda') if device.type=='cuda' else None
    history={'train_loss':[],'val_loss':[]}; start_epoch=0
    if args.resume:
        checkpoint=torch.load(args.resume,map_location=device,weights_only=False)
        model.load_state_dict(checkpoint['model_state_dict']); optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        if checkpoint.get('scheduler_state_dict'): scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        history=checkpoint.get('history',history); start_epoch=checkpoint['epoch']
        log.info('Resumed from epoch %d',start_epoch)
    Path(args.checkpoints).mkdir(parents=True,exist_ok=True); Path(args.output).mkdir(parents=True,exist_ok=True)
    for epoch in range(start_epoch+1,args.epochs+1):
        began=time.time(); tr=train_one_epoch(model,train_loader,optimizer,device,scaler,epoch,args.epochs)
        vl=validation_loss(model,val_loader,device); scheduler.step()
        history['train_loss'].append(tr); history['val_loss'].append(vl)
        log.info('Epoch %d/%d train_loss=%.4f val_loss=%.4f time=%.1f min',epoch,args.epochs,tr,vl,(time.time()-began)/60)
        save_checkpoint(Path(args.checkpoints)/'latest.pth',model,optimizer,scheduler,epoch,train_ds.classes,history)
        if vl <= min(history['val_loss']):
            torch.save({'model_state_dict':model.state_dict(),'class_names':train_ds.classes},Path(args.checkpoints)/'best.pth')
        plot_history(history,args.output)
    torch.save({'model_state_dict':model.state_dict(),'class_names':train_ds.classes},Path(args.checkpoints)/'model.pth')

if __name__=='__main__': main()
