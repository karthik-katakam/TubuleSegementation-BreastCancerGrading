# 🔬 Breast Cancer Grading — Tubule Segmentation

A deep learning pipeline for automated tubule segmentation in breast cancer histopathology whole-slide images (WSI), using a **Recurrent Residual U-Net (R2-UNet)** architecture.

---

## Workflow

```
WSI (.tiff)
    │
    ▼
[1. Tile Extraction]
    Break the whole-slide image into small tiles (e.g. 992×992 px)
    │
    ▼
[2. Data Preprocessing]  ◄── Data_Preprocessing.ipynb
    Organize tiles + masks into Data/<slide_id>/images/ and masks/
    │
    ▼
[3. Training]  ◄── Training.py / Trainwithincodemodel.py / Notebooks
    Resize tiles to 256×256, augment, train R2-UNet
    Save best model weights → model.h5
    │
    ▼
[4. Inference]  ◄── hull.py
    Run model on each tile
    Extract tubule contours as convex hull polygons
    Export global WSI coordinates → JSON annotation file
    │
    ▼
[5. Mask Stitching]  ◄── join.py
    Reassemble per-tile prediction masks
    back into a single full-slide TIFF
```

---

## Project Structure

```
BreastCancerGrading/
├── Data/
│   └── <slide_id>/
│       ├── images/          # Extracted WSI tiles (.jpg)
│       └── masks/           # Corresponding binary segmentation masks (.png)
│
├── utils.py                           # R2-UNet architecture, metrics, helpers
├── Training.py                        # Main training script
├── Trainwithincodemodel.py            # Self-contained training (no utils dependency)
├── hull.py                            # Inference → convex hull polygon → JSON export
├── join.py                            # Stitch tile masks → full-slide TIFF
├── Data_Preprocessing.ipynb          # Tile organization and preprocessing
├── Training_Notebook_-_Tubule.ipynb  # Interactive training notebook
└── Training_with_completedat_Tubule.ipynb  # Training on full dataset
```

---

## Step-by-Step Guide

### Step 1 — Tile Extraction

Start with a `.tiff` WSI file. Break it into fixed-size tiles (992×992 px). Each tile is named using its grid position in the slide:

```
{x}_{y}.jpg     →   e.g.  10_45.jpg  (tile at column 10, row 45)
```

Corresponding masks follow the same naming convention. Any tool can be used for tiling (e.g. QuPath, OpenSlide, pyvips).

---

### Step 2 — Data Preprocessing

Use `Data_Preprocessing.ipynb` to move and organize raw tiles and masks into the expected folder structure:

```
Data/
└── T96/
    ├── images/
    │   ├── T_96X1234Y5678.jpg
    │   └── ...
    └── masks/
        ├── T_96X1234Y5678-mask.png
        └── ...
```

The notebook handled 6,040 image/mask pairs across multiple slides.

---

### Step 3 — Training

The model is trained on 256×256 tiles resized from the 992×992 originals.

**Run training:**

```bash
python Training.py
```

Or the self-contained version (no `utils.py` needed):

```bash
python Trainwithincodemodel.py
```

**Data augmentation:** horizontal flip, vertical flip, and combined flip — effectively 4× the dataset size.

**Train / Val / Test split:** 70% / 10% / 20%

**Key hyperparameters:**

| Parameter | Value |
|---|---|
| Input size | 256 × 256 |
| Learning rate | 2e-4 |
| Batch size | 16 |
| Loss | Binary Crossentropy |
| Metrics | MeanIoU, F1 |
| Output | `model.h5` |

---

### Step 4 — Inference & Polygon Export

`hull.py` runs the trained model over a directory of 992×992 tiles and exports tubule outlines as convex hull polygons with **global WSI coordinates**.

```bash
# Edit the tile directory path and model weights path inside hull.py, then:
python hull.py
```

**What happens internally:**
1. Loads `model.h5`
2. Skips near-blank tiles where `np.std(pixel values) ≤ 12`
3. Predicts a 256×256 binary mask, threshold at 0.4
4. Upscales the mask back to 992×992
5. Finds contours and computes convex hulls (minimum 10 points kept)
6. Translates hull coordinates to global WSI space using the tile's `x_y` filename
7. Writes all polygons to a JSON annotation file

**Output JSON** (compatible with digital pathology viewers):

```json
{
  "name": "NewAlgoV2",
  "elements": [
    {
      "type": "polyline",
      "closed": true,
      "group": "TubuleSegmentation",
      "points": [[x1, y1, 0], [x2, y2, 0], ...],
      "fillColor": "rgba(0,0,0,0)",
      "lineColor": "rgb(0,0,0)",
      "lineWidth": 2
    }
  ]
}
```

---

### Step 5 — Mask Stitching

`join.py` reassembles all per-tile prediction mask PNGs back into a single full-slide TIFF using `pyvips`.

```bash
python join.py
```

Set `tiles_across` and `tiles_down` to match your slide's tile grid dimensions. Expects tiles named `{x}_{y}.png` inside `./17_mask_png/`.

```python
tiles_across = 171
tiles_down   = 155
# Output → t96_mask.tiff
```

---

## Model Architecture — R2-UNet

The segmentation model is a **Recurrent Residual U-Net (R2-UNet)**:

- **Encoder:** 4 downsampling levels with recurrent residual blocks; channels grow 16 → 32 → 64 → 128 → 256
- **Bottleneck:** Recurrent residual block at the deepest level (256 channels)
- **Decoder:** Symmetric upsampling with skip connections and recurrent residual blocks
- **Output:** 1×1 Conv2D → Sigmoid for binary tubule mask

Each recurrent residual block applies two rounds of a two-step recurrent convolution, added back to a 1×1 shortcut projection — capturing fine-grained tubule boundaries.

---

## Setup

### Requirements

```bash
pip install tensorflow opencv-python pillow scikit-learn scikit-image matplotlib pyvips
```

> TensorFlow 2.x. GPU strongly recommended for training.

---

## Results

| | Value |
|---|---|
| Dataset | 6,040 WSI tile / mask pairs |
| Augmented training samples | ~16,900 |
| Model input | 256 × 256 × 3 |
| Inference tile size | 992 × 992 |

*(Add your validation IoU and F1 scores here)*

---

## References

- [R2U-Net — Alom et al., 2018](https://arxiv.org/abs/1802.06955)
- Nottingham Histologic Grading System for breast cancer

---

## License

MIT License.
