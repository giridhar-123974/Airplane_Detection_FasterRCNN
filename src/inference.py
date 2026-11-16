"""Run predictions on one image, many images, or a folder."""
from __future__ import annotations
import argparse
from pathlib import Path
import torch
from PIL import Image
from .model import build_model
from .utils import get_device
from .visualization import draw_detections

def predict(paths, checkpoint_path, output_dir='outputs/predictions', threshold=.5):
    device=get_device(); ckpt=torch.load(checkpoint_path,map_location='cpu',weights_only=False); names=ckpt['class_names']
    model=build_model(len(names),False).to(device); model.load_state_dict(ckpt['model_state_dict']); model.eval()
    output_dir=Path(output_dir); output_dir.mkdir(parents=True,exist_ok=True); results=[]
    for path in paths:
        path=Path(path)
        with Image.open(path) as im: image=im.convert('RGB')
        tensor=torch.from_numpy(__import__('numpy').asarray(image).copy()).permute(2,0,1).float()/255
        with torch.no_grad(): out=model([tensor.to(device)])[0]
        dest=output_dir/f'{path.stem}_prediction.jpg'
        draw_detections(image,out['boxes'].cpu(),out['labels'].cpu(),out['scores'].cpu(),names,dest,threshold)
        results.append({'image':str(path),'output':str(dest),'detections':int((out['scores']>=threshold).sum())})
    return results

def main():
    p=argparse.ArgumentParser(description=__doc__); p.add_argument('input'); p.add_argument('--checkpoint',default='checkpoints/best.pth'); p.add_argument('--output',default='outputs/predictions'); p.add_argument('--threshold',type=float,default=.5); a=p.parse_args()
    path=Path(a.input); paths=sorted(x for x in path.iterdir() if x.suffix.lower() in {'.jpg','.jpeg','.png','.bmp'}) if path.is_dir() else [path]
    for result in predict(paths,a.checkpoint,a.output,a.threshold): print(f"{result['output']}: {result['detections']} detections")

if __name__=='__main__': main()
