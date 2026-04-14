# YOLOv10n Weapon Detection Model

## Overview

This repository contains a fine-tuned YOLOv10n model for weapon detection based on the SOHAS dataset.

The model is optimized for lightweight deployment and is suitable for edge devices such as Raspberry Pi.

----

## Model Information

| Property | Value |
|----------|-------|
| **Model** | YOLOv10n |
| **Input Size** | 320 × 320 |
| **Framework** | Ultralytics YOLO |
| **Task** | Object Detection |

----

## Classes

| Class | Description |
|-------|-------------|
| **knife** | Knife |
| **pistol** | Pistol/Gun |
| **smartphone** | Smartphone |
| **money** | Paper Money/Bill |
| **card** | Credit Card |
| **other** | Other objects (wallet, purse, etc.) |

----

## Validation Performance

| Metric | Value |
|--------|-------|
| **mAP@0.5** | 83.94% |
| **mAP@0.5:0.95** | 70.23% |
| **Precision** | 89.49% |
| **Recall** | 83.02% |

----

## Files

### Weights

- `weights/best.pt` — PyTorch model (best validation performance)
- `weights/model.onnx` — ONNX model for deployment

### Config

- `config/data.yaml` — Dataset configuration

### Results

- Confusion Matrix
- Precision-Recall Curve
- F1 Curve

### Samples

- Example inference outputs

----

## Inference (PyTorch)

```python
from ultralytics import YOLO

model = YOLO('weights/best.pt')
results = model('test.jpg', conf=0.25)

for r in results:
    r.show()
```

## Inference (ONNX)

```python
import cv2
import numpy as np
import onnxruntime as ort

session = ort.InferenceSession("weights/model.onnx", providers=['CPUExecutionProvider'])
input_name = session.get_inputs()[0].name

img = cv2.imread("test.jpg")
img = cv2.resize(img, (320, 320))
img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
img = np.transpose(img, (2, 0, 1))
img = np.expand_dims(img, axis=0)

outputs = session.run(None, {input_name: img})
```

----

## Deployment

This model is designed for deployment on low-resource environments such as Raspberry Pi using ONNX Runtime.

### Raspberry Pi Setup

1. Install dependencies:
```bash
pip install opencv-python numpy onnxruntime
```

2. Copy model files:
```bash
scp weights/model.onnx pi@raspberry-ip:/home/pi/model/
```

3. Run inference:
```bash
python3 inference.py --image test.jpg --model model.onnx
```

----

## Notes

- The model is trained on the SOHAS Weapon Detection dataset.
- Input resolution is fixed at 320 for efficient inference.
- This model is intended for research and educational purposes. Ensure proper authorization before using in production environments.
