"""Plot training history and detection results."""
from pathlib import Path
import matplotlib.pyplot as plt
import matplotlib.patches as patches

def plot_history(history, output_dir):
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(history['train_loss'], label='Train loss')
    if history.get('val_loss'): ax.plot(history['val_loss'], label='Validation loss')
    ax.set(xlabel='Epoch', ylabel='Loss', title='Faster R-CNN training'); ax.grid(alpha=.25); ax.legend()
    fig.tight_layout(); fig.savefig(Path(output_dir) / 'loss.png', dpi=160); plt.close(fig)

def draw_detections(image, boxes, labels, scores, class_names, output_path=None, threshold=.5, title=None):
    fig, ax = plt.subplots(figsize=(10, 10)); ax.imshow(image); ax.axis('off')
    for box, label, score in zip(boxes, labels, scores):
        if float(score) < threshold: continue
        x1, y1, x2, y2 = map(float, box)
        ax.add_patch(patches.Rectangle((x1,y1),x2-x1,y2-y1,fill=False,edgecolor='lime',linewidth=2))
        name = class_names[int(label)-1] if int(label)>0 and int(label)<=len(class_names) else str(label)
        ax.text(x1, max(0,y1-4), f'{name} {float(score):.2f}', color='black', fontsize=9,
                bbox={'facecolor':'lime','alpha':.8,'pad':2})
    if title: ax.set_title(title)
    fig.tight_layout(pad=0)
    if output_path:
        Path(output_path).parent.mkdir(parents=True, exist_ok=True); fig.savefig(output_path, dpi=160, bbox_inches='tight',pad_inches=0)
    return fig
