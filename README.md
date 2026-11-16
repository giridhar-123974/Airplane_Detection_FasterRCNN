# Airplane Detection with Faster R-CNN

A modular PyTorch/TorchVision object detection pipeline for the Airbus Aircrafts Detection Sample Dataset. It reads the dataset's verified polygon CSV format and converts each polygon into an axis-aligned pixel bounding box.

## Dataset findings

The supplied dataset README describes 103 annotated JPEG satellite image crops (2560×2560) plus six unannotated images in `extras/`. The CSV has 3,425 rows and columns `id`, `image_id`, `geometry`, and `class`. `geometry` is a closed polygon represented as a Python-style list of `(x, y)` pixel points. Image coordinates begin at the upper-left; each polygon is reduced to `[min_x, min_y, max_x, max_y]`. Labels found in the actual CSV are `Airplane` and `Truncated_airplane` (3,316 and 109 rows). The model maps these to IDs 1 and 2; ID 0 is background. There are no duplicate annotations or annotation image names missing from `images/` in the supplied copy.

Splits are made at image level, with a deterministic seed, and saved to `dataset/splits/`. This avoids leaking objects from the same source image across training and validation. Invalid polygon rows are warned about and skipped; unreadable images fail with a path-specific error rather than silently disappearing from the dataset.

The dataset is distributed under CC BY-NC-SA 4.0. The dataset README identifies the curator as jeffaudi and Airbus DS Intelligence as the imagery provider. Preserve attribution and license terms when redistributing dataset contents or adaptations. See `dataset/README.md` and `dataset/LICENSE.txt`.

## Requirements and installation

Python 3.11+ is recommended. Install a PyTorch and TorchVision pair appropriate for your CUDA runtime first, using the official PyTorch installation selector, then install the remaining dependencies:

```bash
pip install -r requirements.txt
```

On a CUDA machine, confirm `torch.cuda.is_available()` before training. CPU training works but is slow for 2560-pixel images.

## Project structure

```text
dataset/                 Supplied images, annotations, license, and extras
src/dataset.py           CSV validation, polygon parsing, dataset and splits
src/transforms.py        Tensor conversion and horizontal flip
src/model.py             Faster R-CNN model factory
src/engine.py             Training and validation-loss loops
src/train.py              Training CLI and checkpointing
src/evaluate.py           Class-wise AP at IoU 0.5
src/inference.py          Single image or folder inference CLI
src/visualization.py      Loss plot and detection overlays
src/utils.py               Device, seed, logging, checkpoint helpers
notebooks/FasterRCNN_Training.ipynb  Colab walkthrough
checkpoints/              Generated checkpoints (ignored by Git)
outputs/                  Generated plots and predictions
models/                   Optional exported model location
```

## Training

From the repository root:

```bash
python -m src.train --dataset dataset --epochs 20 --batch-size 2
```

The default uses COCO pretrained Faster R-CNN weights, replaces the classifier for the two dataset labels plus background, and trains with SGD. CUDA mixed precision is enabled when CUDA is available. Adjust batch size to GPU memory; high-resolution images can require batch size 1. Epoch checkpoints are saved to `checkpoints/latest.pth`, the best validation-loss weights to `checkpoints/best.pth`, and final weights to `checkpoints/model.pth`. Training loss and validation loss are plotted to `outputs/loss.png`.

Resume from the latest full-state checkpoint:

```bash
python -m src.train --dataset dataset --epochs 40 --resume checkpoints/latest.pth
```

`--epochs` is the final epoch count, so use a value higher than the checkpoint epoch. The generated CSV splits persist and can be regenerated with `create_splits(..., overwrite=True)`.

## Evaluation

```bash
python -m src.evaluate --dataset dataset --checkpoint checkpoints/best.pth
```

Evaluation reports per-class AP and mAP at IoU 0.5, calculated from ranked detections and one-to-one ground-truth matching on the validation images. mAP is therefore available for this bounding-box dataset. The class `Truncated_airplane` is kept separate because that is how it is labeled in the source CSV.

## Inference

```bash
python -m src.inference dataset/extras --checkpoint checkpoints/best.pth
python -m src.inference path/to/image.jpg --checkpoint checkpoints/best.pth --threshold 0.4
```

Rendered images with labels and confidence scores are saved under `outputs/predictions/`.

## Google Colab: exact Run all steps

The notebook is configured for the fixed Google Drive layout below and needs no code edits:

```text
MyDrive/Airplane_Detection_FasterRCNN/
├── dataset/
│   ├── annotations.csv
│   ├── images/
│   └── extras/
├── src/
├── requirements.txt
└── notebooks/FasterRCNN_Training.ipynb
```

1. Copy the complete project folder to `MyDrive/Airplane_Detection_FasterRCNN/`, keeping the supplied dataset in its `dataset/` subfolder.
2. In Google Colab, open `notebooks/FasterRCNN_Training.ipynb` from that Drive project folder. Set the Colab runtime to a GPU for practical training speed; the code also runs on CPU, more slowly.
3. Choose **Runtime → Run all**. The first cell mounts Google Drive and asks you to authorize access. The notebook installs `requirements.txt` dependencies, checks required CSV columns, missing files, image readability, polygon validity, and availability of at least five `extras/` images before training.
4. The notebook trains for 20 epochs at batch size 1, evaluates the held-out validation split, and runs inference over the unannotated `extras/` folder. It displays five prediction images with bounding boxes, labels, and confidence scores, along with class AP and mAP at IoU 0.5 and the loss plot.

The best weights are saved as `MyDrive/Airplane_Detection_FasterRCNN/best_model.pth`. Training checkpoints, loss plot, and rendered predictions are saved under the same Drive project folder in `checkpoints/` and `outputs/`. The Drive mount requires the normal Google Colab authorization step; no dataset path or code changes are needed.

## Expected outputs

- `dataset/splits/train.csv`, `dataset/splits/val.csv`
- `checkpoints/latest.pth`, `best.pth`, and `model.pth`; the Colab notebook also writes `best_model.pth` in the project root on Drive
- `outputs/loss.png`
- `outputs/predictions/*_prediction.jpg`

## Troubleshooting

- **CUDA out of memory:** set `--batch-size 1`; if necessary, use a smaller image copy or a GPU with more memory. The supplied images are 2560×2560.
- **Torch/TorchVision operator mismatch:** reinstall a matched pair from the official PyTorch instructions for your CUDA or CPU runtime.
- **No matching images:** `image_id` in the CSV must match filenames in the selected `images/` directory. For other schemas, update the explicit schema validation and polygon conversion instead of guessing.
- **No pretrained download:** the model factory logs a warning and falls back to random initialization if fetching pretrained weights fails; use `--no-pretrained` to choose that behavior explicitly.
- **Validation AP seems low:** inspect the saved detections and verify that training has converged; this is a small demonstration subset of satellite imagery.
