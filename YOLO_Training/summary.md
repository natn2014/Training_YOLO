# YOLO Training Tool - User Manual & Working Flow

**Target User:** Production Engineers  
**Version:** 1.0  
**Last Updated:** February 2, 2026

---

## Table of Contents
1. [Overview](#overview)
2. [System Requirements](#system-requirements)
3. [Application Structure](#application-structure)
4. [Complete Workflow](#complete-workflow)
5. [Tab-by-Tab Guide](#tab-by-tab-guide)
6. [Key Features](#key-features)
7. [Troubleshooting](#troubleshooting)
8. [Tips & Best Practices](#tips--best-practices)

---

## Overview

The **YOLO Training Tool** is a comprehensive graphical interface for training custom YOLO object detection models. It simplifies the entire workflow from dataset preparation through model deployment, eliminating the need for command-line operations.

### Key Capabilities:
- Dataset validation and visualization
- Multiple pre-trained model selection (YOLOv11, YOLOv8, YOLOv10)
- Customizable training parameters
- Real-time training monitoring with live charts
- Model testing and validation
- Export to multiple formats (ONNX, TensorRT, TFLite)

---

## System Requirements

### Hardware:
- **GPU:** NVIDIA GPU recommended (RTX 3060 or better for faster training)
- **RAM:** Minimum 8GB (16GB+ recommended)
- **Storage:** 50GB+ free space for models and datasets

### Software:
- Python 3.8 or later
- CUDA 11.8+ (if using GPU)
- Dependencies: See `requirements.txt`

### Installation:
```bash
pip install -r requirements.txt
```

### Run the Application:
**Windows:**
```bash
python yolo_trainer_gui.py
```

Or use the batch script:
```bash
run_gui.bat
```

**Linux/Mac:**
```bash
python yolo_trainer_gui.py
```

Or use the shell script:
```bash
bash run_gui.sh
```

---

## Application Structure

The application uses a **5-tab workflow** design:

```
┌─────────────────────────────────────────────────────┐
│  YOLO Training Tool - Production Engineer Edition   │
├─────────────────────────────────────────────────────┤
│ ┌─────────────────────────────────────────────────┐ │
│ │ Tab 1: Dataset  │ Tab 2: Model Selection │ Tab 3: Configuration │ Tab 4: Training │ Tab 5: Results & Export │
│ └─────────────────────────────────────────────────┘ │
└─────────────────────────────────────────────────────┘
```

---

## Complete Workflow

### Quick Start (5 Steps):

```
1. PREPARE DATASET
   └─> Tab 1: Select data.yaml and validate dataset

2. SELECT MODEL
   └─> Tab 2: Choose pre-trained model (YOLOv11 recommended)

3. CONFIGURE TRAINING
   └─> Tab 3: Set epochs, batch size, learning rate, etc.

4. START TRAINING
   └─> Tab 4: Click "Start Training" and monitor progress

5. EVALUATE & EXPORT
   └─> Tab 5: Test on images, validate, export to production format
```

---

## Tab-by-Tab Guide

### **Tab 1: Dataset Selection & Validation**

**Purpose:** Configure and validate your training dataset

#### Dataset Structure Required:
```
dataset_root/
├── train/
│   ├── images/       (training images)
│   └── labels/       (YOLO format .txt files)
├── val/
│   ├── images/       (validation images)
│   └── labels/       (validation labels)
└── test/ (optional)
    ├── images/
    └── labels/
```

#### Label File Format (YOLO Format):
Each `.txt` file should contain one object per line:
```
<class_id> <x_center> <y_center> <width> <height>
```
All coordinates are normalized (0-1 range).

#### Steps:
1. **Select Dataset Folder:** Click "Browse..." to choose your dataset root directory
2. **Auto-detect data.yaml:** The tool automatically searches for `data.yaml` in the selected folder
3. **Or Manually Select data.yaml:** Click "Browse..." under "data.yaml File"
4. **Validate Dataset:** Click "Validate Dataset" button

#### What Gets Validated:
- ✓ Dataset structure correctness
- ✓ Image count vs. label count matching
- ✓ Total object distribution
- ✓ Class names and count

#### Visualization:
- **Bar Chart:** Shows train/validation/test split distribution
- **Pie Chart:** Displays class distribution with actual object counts from all label files

#### Sample data.yaml Format:
```yaml
path: /path/to/dataset
train: train/images
val: val/images
test: test/images

nc: 2                    # number of classes
names: ['class1', 'class2']  # class names
```

---

### **Tab 2: Model Selection**

**Purpose:** Choose the pre-trained model for transfer learning

#### Available Models:

**YOLOv11 (Recommended for 2025+):**
- `yolo11n.pt` - Nano (Fastest, least accurate)
- `yolo11s.pt` - Small (Balanced)
- `yolo11m.pt` - Medium (Good accuracy)
- `yolo11l.pt` - Large (High accuracy)
- `yolo11x.pt` - XLarge (Most accurate, slowest)

**YOLOv8 (Stable, proven):**
- `yolov8n.pt` through `yolov8x.pt`

**YOLOv10:**
- `yolov10n.pt` through `yolov10x.pt`

**Custom Model:**
- Browse and select your own `.pt` file

#### Performance Comparison:
The tool displays a table with:
- mAP50-95 (accuracy on COCO dataset)
- Speed (inference time in ms)
- Parameters (model size)
- File Size (download size)

#### Recommendations by Use Case:

| Use Case | Recommended Model | Reason |
|----------|------------------|--------|
| Edge devices (Jetson Orin Nano) | yolo11n or yolo11s | Low memory, real-time speed (~80 FPS) |
| Production accuracy | yolo11m or yolo11l | Balanced speed and accuracy |
| Research/Maximum accuracy | yolo11x | Best accuracy, requires powerful GPU |
| Lightweight mobile | yolo11n | Smallest size (~5MB) |

---

### **Tab 3: Training Configuration**

**Purpose:** Set all training hyperparameters

#### Basic Training Parameters:

| Parameter | Default | Range | Description |
|-----------|---------|-------|-------------|
| Epochs | 100 | 1-1000 | Number of training iterations |
| Image Size | 640 | 320, 416, 512, 640, 800, 1024, 1280 | Input resolution (larger = better accuracy but slower) |
| Batch Size | 16 | -1 to 128 | Images per batch (-1 = auto) |
| Device | 0 (GPU 0) | 0, 1, cpu | Computing device |
| Patience | 50 | 0-200 | Early stopping (epochs without improvement) |
| Workers | 8 | 0-32 | Data loading threads |

#### Optimization Parameters:

| Parameter | Default | Description |
|-----------|---------|-------------|
| Optimizer | auto | SGD, Adam, AdamW, NAdam, RAdam, RMSProp |
| Learning Rate | 0.01 | Initial learning rate |
| Weight Decay | 0.0005 | L2 regularization |
| Warmup Epochs | 3.0 | Learning rate warmup |

#### Data Augmentation:

| Parameter | Default | Range | Effect |
|-----------|---------|-------|--------|
| Flip Left-Right | 0.5 | 0.0-1.0 | Horizontal flip probability |
| Mosaic | 1.0 | 0.0-1.0 | Mosaic augmentation probability |
| HSV Hue | 0.015 | 0.0-0.1 | Color hue variation |
| Scale | 0.5 | 0.0-1.0 | Image scale variation |

#### Output Settings:

- **Project Folder:** Where training outputs are saved
  - Default: `{current_directory}/runs/train`
  - All results use absolute paths for clarity

- **Experiment Name:** Auto-generated with timestamp
  - Format: `exp_YYYYMMDD_HHMMSS`
  - Used to organize multiple training runs

- **Mixed Precision (AMP):** Recommended for GPU training (faster, less memory)
- **Generate Plots:** Creates training visualization plots

#### Quick Configuration Presets:

**For Fast Training (Edge Deployment):**
```
- Epochs: 50
- Batch Size: 32
- Image Size: 416
- Optimizer: SGD
```

**For Production Quality:**
```
- Epochs: 100
- Batch Size: 16
- Image Size: 640
- Optimizer: AdamW
- Learning Rate: 0.001
```

**For Research/Maximum Accuracy:**
```
- Epochs: 300
- Batch Size: 8
- Image Size: 1280
- Optimizer: AdamW
- Learning Rate: 0.0001
```

---

### **Tab 4: Training Execution & Monitoring**

**Purpose:** Train your model and monitor progress in real-time

#### Starting Training:

1. **Validate Prerequisites:**
   - Dataset must be configured (Tab 1)
   - Model must be selected (Tab 2)
   - Training config must be set (Tab 3)

2. **Click "Start Training"**
   - System validates all inputs
   - Model auto-downloads if not local (first time only)
   - Training begins in background thread (UI remains responsive)

#### Real-Time Monitoring:

**Progress Bar:**
- Shows current epoch progress
- Displays percentage completion

**Training Metrics:**
- Epoch: Current epoch / Total epochs
- Box Loss: Localization loss
- Cls Loss: Classification loss
- Precision: True positive rate
- Recall: Detection sensitivity
- mAP50: Average precision at IoU 0.5

**Live Charts:**
Three synchronized charts update every epoch:
1. **Loss Chart:** Box loss and classification loss trends
2. **Precision & Recall:** Model detection metrics
3. **mAP Chart:** mAP50 and mAP50-95 scores

**Training Log:**
- Timestamped messages
- Key milestones and updates
- Auto-scrolls to latest message

#### During Training:

- Model can be paused (click "Stop Training")
- Cannot close window while training without confirmation
- GPU utilization visible in system monitor
- Memory usage automatically managed

#### After Training:

- Training complete notification
- Best model path displayed
- Results automatically saved to project folder
- Model auto-loaded in Results tab

---

### **Tab 5: Results & Export**

**Purpose:** Test, validate, and deploy your trained model

#### Model Path Management:

- **Browse:** Select any trained `.pt` model file
- **Auto-populated:** Best model from training is auto-loaded
- Pre-trained models can be tested before training

#### Testing on Single Image:

1. **Select Model:** Model must be set first
2. **Click "Test on Image":**
   - Choose an image file (JPG, PNG, BMP, WebP)
   - Model runs inference
   - Predictions saved with bounding boxes

3. **View Prediction Result:**
   - Predicted image displays in "Prediction Result" panel
   - Shows detected objects with confidence scores
   - Full resolution image accessible via scroll

#### Model Validation:

1. **Click "Validate Model":**
   - Runs inference on entire validation dataset
   - Computes metrics: mAP50, mAP50-95, Precision, Recall

2. **Results Table:**
   - mAP50: Accuracy at IoU threshold 0.5
   - mAP50-95: Accuracy averaged across IoU 0.5-0.95
   - Precision: Positive prediction accuracy
   - Recall: Detection coverage
   - F1 Score: Harmonic mean of precision and recall

#### Model Export:

**Export Formats:**

| Format | Use Case | Benefits | Drawbacks |
|--------|----------|----------|-----------|
| **ONNX** | Cross-platform | Universal standard | Requires runtime |
| **TensorRT (Engine)** | NVIDIA GPU deployment | Fastest on NVIDIA hardware | GPU-specific |
| **TFLite** | Mobile/Edge devices | Lightweight | Lower accuracy sometimes |

**Export Options:**
- **Image Size:** Resolution for optimization (640 default)
- **Precision:** FP16 (Half) for speed/memory vs FP32 (Full) for accuracy

**Export Steps:**
1. Select model
2. Choose export format
3. Set image size and precision
4. Click export button
5. Wait for completion
6. Exported model ready for deployment

#### Export Location:
All exports saved to: `{project_folder}/{experiment_name}/`

---

## Key Features

### 1. **Automatic Model Download**
- Pre-trained models auto-download on first use
- Cached for subsequent training
- Works offline after first download

### 2. **Path Management**
- All paths converted to absolute paths automatically
- No relative path issues
- Consistent across different working directories

### 3. **Real-Time Live Charts**
- Charts update every epoch
- Multiple metrics tracked simultaneously
- Helps detect training issues early

### 4. **Class Distribution Visualization**
- Actual object counts from label files
- Helps identify class imbalance
- Pie chart auto-scales for readability

### 5. **Callback System**
- Metrics registered via callbacks
- Live chart updates from model callbacks
- Ensures accurate real-time monitoring

### 6. **Error Handling**
- Version compatibility checks
- Model download fallback logic
- Detailed error messages
- Graceful error recovery

### 7. **Threading**
- Training runs on separate thread
- UI remains responsive
- Can't accidentally freeze interface

---

## Troubleshooting

### Common Issues & Solutions

#### Issue: "Model not found" error
**Cause:** Pre-trained model doesn't exist locally  
**Solution:** 
- Check internet connection
- Model will auto-download
- Wait for download to complete

#### Issue: Out of Memory (OOM) error
**Solution:**
- Reduce batch size (16 → 8 or -1 for auto)
- Reduce image size (640 → 416 or 512)
- Use smaller model (yolo11n instead of yolo11m)
- Enable AMP (Mixed Precision)

#### Issue: Slow training speed
**Cause:** Using CPU or wrong device  
**Solution:**
- Check device selection (should show "0 (GPU 0)" not "cpu")
- Verify GPU is available: `nvidia-smi`
- Update NVIDIA drivers
- Check GPU isn't used by other processes

#### Issue: Poor model accuracy
**Cause:** Dataset issues or insufficient training  
**Solution:**
- Validate dataset (check class balance)
- Increase epochs
- Use larger model (yolo11m/l)
- Better image quality
- More diverse training images

#### Issue: Metrics not updating in Tab 4
**Cause:** Callback registration issue  
**Solution:**
- Restart training
- Check that callbacks are registered before training starts
- Verify ultralytics version >= 8.1.0

#### Issue: Predicted image not showing
**Cause:** Image path or format issue  
**Solution:**
- Use standard format (JPG, PNG)
- Ensure image dimensions match model expectations
- Check file has read permissions

#### Issue: "Can't get attribute 'C3k2'" error
**Cause:** YOLOv11 incompatibility with older ultralytics  
**Solution:**
```bash
pip install --upgrade ultralytics>=8.1.0
```

---

## Tips & Best Practices

### Dataset Preparation:
- ✓ Ensure consistent image quality
- ✓ Include diverse lighting conditions
- ✓ Maintain 70/20/10 split (train/val/test)
- ✓ Balance classes (avoid too many of one class)
- ✓ Use high-quality annotations

### Training Tips:
- ✓ Start with smaller model for quick iteration
- ✓ Use learning rate scheduler
- ✓ Monitor loss curves for overfitting
- ✓ Set appropriate patience for early stopping
- ✓ Use data augmentation for small datasets

### Model Selection:
- ✓ Start with yolo11n for prototyping
- ✓ Move to yolo11s for production
- ✓ Use yolo11m/l only if accuracy critical
- ✓ Match model size to deployment hardware

### Performance Optimization:
- ✓ Increase batch size when memory allows
- ✓ Use AMP for faster training
- ✓ Reduce image size for edge devices
- ✓ Export to TensorRT for NVIDIA deployment

### Reproducibility:
- ✓ Note all hyperparameters
- ✓ Use same random seed if needed
- ✓ Archive trained models and configs
- ✓ Keep experiment names descriptive

### Monitoring:
- ✓ Check training log for anomalies
- ✓ Watch loss curves for divergence
- ✓ Monitor GPU memory usage
- ✓ Validate periodically on test set

---

## Advanced Usage

### Multi-GPU Training:
Currently uses single GPU. For multi-GPU, modify:
```python
device=0  # Change to [0,1] for GPUs 0 and 1
```

### Custom Model Files:
1. Go to Tab 2: Model Selection
2. Select "Custom .PT File"
3. Browse to your `.pt` model file
4. Proceed with training

### Custom Dataset Formats:
Tool uses YOLO format. To convert:
- COCO format → YOLO: Use ultralytics utilities
- VOC format → YOLO: Use conversion scripts

### Export for Jetson:
1. Train on full-size GPU
2. Export to ONNX or TensorRT
3. Deploy on Jetson Orin Nano

---

## File Structure

```
ONNX_Training/
├── yolo_trainer_gui.py       # Main application
├── ui_styles.py              # Theme and styling
├── requirements.txt          # Python dependencies
├── data.yaml.example         # Example dataset config
├── DATASET_STRUCTURE_GUIDE.md # Detailed dataset guide
├── summary.md                # This file
├── run_gui.bat              # Windows launcher
├── run_gui.sh               # Linux/Mac launcher
├── runs/                    # Training outputs (auto-created)
│   └── train/
│       └── exp_YYYYMMDD_HHMMSS/
│           ├── weights/
│           │   ├── best.pt
│           │   ├── last.pt
│           │   └── ...
│           ├── plots/
│           └── ...
└── README.md                # Project overview
```

---

## Support & Resources

### Documentation:
- [Ultralytics YOLOv11 Docs](https://docs.ultralytics.com/)
- [YOLO Format Guide](https://roboflow.com/formats/yolo-darknet-txt)
- DATASET_STRUCTURE_GUIDE.md (in project)

### Common Questions:

**Q: How long does training take?**  
A: 1-48 hours depending on model size, dataset size, and GPU. Start with yolo11n for faster iteration.

**Q: Can I resume training?**  
A: Load the `last.pt` from previous run and train with same config.

**Q: What's the difference between best.pt and last.pt?**  
A: `best.pt` = highest validation accuracy; `last.pt` = final epoch weights

**Q: Can I use this on CPU?**  
A: Yes, but 10-100x slower than GPU. Not recommended for production.

---

## Version History

| Version | Date | Changes |
|---------|------|---------|
| 1.0 | 2026-02-02 | Initial release |

---

## License & Attribution

Built with:
- **Ultralytics YOLOv11** - Object detection framework
- **PySide6** - Qt6 Python bindings
- **Matplotlib** - Data visualization

---

**Last Updated:** February 2, 2026  
**Status:** Production Ready
