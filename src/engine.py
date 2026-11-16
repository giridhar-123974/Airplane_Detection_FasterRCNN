"""Training and validation loops."""
from __future__ import annotations
import logging, time
import torch
from tqdm.auto import tqdm

LOGGER = logging.getLogger(__name__)

def train_one_epoch(model, loader, optimizer, device, scaler=None, epoch=1, total_epochs=1):
    model.train(); running = 0.; start = time.time()
    bar = tqdm(loader, desc=f'Epoch {epoch}/{total_epochs}', leave=False)
    for step, (images, targets) in enumerate(bar, 1):
        images = [image.to(device) for image in images]
        targets = [{k:v.to(device) for k,v in t.items()} for t in targets]
        optimizer.zero_grad(set_to_none=True)
        with torch.autocast(device_type=device.type, enabled=(scaler is not None)):
            losses = model(images, targets); loss = sum(losses.values())
        if not torch.isfinite(loss): raise RuntimeError(f'Non-finite loss: {loss.item()}')
        if scaler:
            scaler.scale(loss).backward(); scaler.step(optimizer); scaler.update()
        else:
            loss.backward(); optimizer.step()
        running += float(loss.detach()); avg = running / step
        elapsed = time.time() - start; eta = elapsed / step * (len(loader)-step)
        bar.set_postfix(loss=f'{avg:.4f}', eta=f'{eta/60:.1f}m')
    return running / max(1,len(loader))

@torch.no_grad()
def validation_loss(model, loader, device):
    """Compute detector losses while keeping normalization layers in eval mode."""
    model.eval()
    # TorchVision detection models return losses only in training mode.
    model.train()
    for module in model.modules():
        if isinstance(module, torch.nn.modules.batchnorm._BatchNorm): module.eval()
    running=0.; count=0
    for images, targets in tqdm(loader, desc='Validation loss', leave=False):
        images=[x.to(device) for x in images]
        targets=[{k:v.to(device) for k,v in t.items()} for t in targets]
        losses=model(images, targets); running += float(sum(losses.values())); count += 1
    model.eval()
    return running / max(1,count)
