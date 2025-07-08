# Image Segmentation: Classical vs Deep Learning Approaches

---

This project presents a comparative study between two binary semantic segmentation approaches applied to the Oxford-IIIT Pet Dataset:  
(1) a classical approach based on OpenCV's GrabCut algorithm, and  
(2) a convolutional encoder-decoder architecture implemented in PyTorch.

---

## Requirements

- Python 3.8+
- PyTorch
- OpenCV
- torchvision
- matplotlib
- tqdm

Install the dependencies:

```bash
pip install -r requirements.txt
```

---

## How to Run

### 1.Train Neural Network

```bash
python methods/deep.py --mode train --plot
```

This will save :

- the model in `models/trained/unet.pth`
- the training metrics in `outputs/training/`

---

### 2. Run U-Net Prediction

```bash
python methods/deep.py --mode predict
```

This will save the segmentation masks in `outputs/unet/`

---

### 5. Run GrabCut

```bash
python methods/classical.py --all
```

This will save the segmentation masks in `outputs/classical/`

---

### 6.Evaluate Results

```bash
python evaluation/evaluate.py
```

This will show the metrics for each model and save the comparative histograms in:

- `outputs/classical_metrics/`
- `outputs/unet_metrics/`

---

### 7. Visual Comparison

```bash
python scripts/visualize_examples.py
```

Saves the figure in:  
`outputs/visualize/visualize.pdf`

---

## Project Structure

```
datasets/         # Dataset loader
models/           # encoder -decoder model
methods/          # Training and classical segmentation scripts
utils/            # Metrics
evaluation/       # Evaluation pipeline
scripts/          # Visual comparison
```

---
