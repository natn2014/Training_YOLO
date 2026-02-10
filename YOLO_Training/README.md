# YOLO Training GUI - Production Engineer Edition

A user-friendly graphical interface for training custom YOLO object detection models. Designed specifically for production engineers with an intuitive multi-tab workflow.

## Features

### 🎯 Multi-Tab Interface
- **Tab 1: Dataset** - Select and validate your dataset with visual statistics
- **Tab 2: Model Selection** - Choose pre-trained YOLO models or load custom models
- **Tab 3: Configuration** - Configure all training parameters with intuitive controls
- **Tab 4: Training** - Monitor training progress with live charts and metrics
- **Tab 5: Results & Export** - Test, validate, and export your trained model

### 📊 Live Visualization
- Real-time training metrics charts (Loss, Precision, Recall, mAP)
- Dataset distribution visualization (train/val/test splits)
- Class distribution charts
- Progress bars and status indicators

### 🚀 Key Features
- **No Manual Typing** - File dialogs for all file/folder selections
- **Dropdown Menus** - Easy selection for all configuration options
- **Threaded Training** - UI remains responsive during training
- **Live Metrics** - Real-time updates of training progress
- **Export Options** - Convert to ONNX, TensorRT, TFLite formats
- **Model Testing** - Quick inference testing on images
- **Validation** - Comprehensive model validation with metrics

## Installation

### 1. Clone or Download

```bash
cd d:\Dev_AI_env\YOLO_TRAINING
```

### 2. Install Dependencies

```bash
pip install -r requirements.txt
```

**Note for Jetson Orin Nano users:**
```bash
# JetPack already includes PyTorch, so install only:
pip install PySide6 ultralytics matplotlib pyyaml
```

### 3. Verify Installation

```bash
python -c "from PySide6.QtWidgets import QApplication; from ultralytics import YOLO; print('OK')"
```

## Quick Start

### Step 1: Launch Application

```bash
python yolo_trainer_gui.py
```

### Step 2: Prepare Dataset

1. Go to **"1. Dataset"** tab
2. Click **"Browse..."** to select your dataset folder
3. The dataset should have this structure:
```
my_dataset/
├── images/
│   ├── train/
│   ├── val/
│   └── test/ (optional)
├── labels/
│   ├── train/
│   ├── val/
│   └── test/ (optional)
└── data.yaml
```

4. Create `data.yaml` file:
```yaml
path: /absolute/path/to/my_dataset
train: images/train
val: images/val
test: images/test  # optional

names:
  0: person
  1: car
  2: bicycle
  # ... add your classes

nc: 3  # number of classes
```

5. Click **"Validate Dataset"** to check everything is correct

### Step 3: Select Model

1. Go to **"2. Model Selection"** tab
2. Choose **"Model Source"** from dropdown:
   - YOLOv11 Pre-trained (Recommended)
   - YOLOv8 Pre-trained
   - YOLOv10 Pre-trained
   - Custom .PT File
3. Select model size:
   - **yolo11n.pt** - Fastest (recommended for Jetson Orin Nano)
   - **yolo11s.pt** - Balanced
   - **yolo11m.pt** - More accurate
   - **yolo11l/x.pt** - Most accurate (slower)

### Step 4: Configure Training

1. Go to **"3. Configuration"** tab
2. Set basic parameters:
   - **Epochs:** 100-300 (default: 100)
   - **Image Size:** 640 (standard)
   - **Batch Size:** 16 (adjust based on GPU memory)
   - **Device:** 0 (GPU 0) or cpu
   - **Patience:** 50 (early stopping)

3. Adjust optimization (optional):
   - **Optimizer:** auto (recommended)
   - **Learning Rate:** 0.01 (default)
   - **Weight Decay:** 0.0005

4. Configure augmentation (optional)
5. Set output folder and experiment name

### Step 5: Start Training

1. Go to **"4. Training"** tab
2. Click **"Start Training"** button
3. Monitor live metrics:
   - Progress bar shows overall progress
   - Live charts update each epoch
   - Training log shows detailed messages
   - Current metrics displayed in real-time

4. Training runs in background thread - UI remains responsive
5. Click **"Stop Training"** if needed

### Step 6: Results & Export

1. Go to **"5. Results & Export"** tab
2. Trained model path appears automatically
3. Available actions:
   - **Test on Image** - Quick inference test
   - **Validate Model** - Full validation with metrics
   - **Export to ONNX** - For cross-platform deployment
   - **Export to TensorRT** - For Jetson/NVIDIA GPUs
   - **Export to TFLite** - For mobile devices

## Dataset Preparation Tools

### Using LabelImg (Manual Annotation)
```bash
pip install labelImg
labelImg
```

### Using Roboflow (Online)
1. Visit https://roboflow.com
2. Upload images
3. Annotate online
4. Export in YOLO format

### Using Label Studio (Running in your terminal)
```bash
# Already running in terminal "label-studio"
# Access at http://localhost:8080
```

## Model Performance Guide

### For Jetson Orin Nano

| Model | Speed (FPS) | Accuracy | Memory | Best Use Case |
|-------|-------------|----------|--------|---------------|
| yolo11n | ~80 | Good | Low | Real-time, edge devices |
| yolo11s | ~50 | Better | Medium | Balanced production |
| yolo11m | ~30 | High | High | Quality inspection |
| yolo11l/x | ~15-20 | Highest | Very High | Maximum accuracy |

**Recommendation:** Start with **yolo11n** or **yolo11s** for Jetson Orin Nano

## Troubleshooting

### Issue: CUDA Out of Memory
**Solution:** Reduce batch size in Tab 3 (try 8 or 4)

### Issue: Training Too Slow
**Solution:** 
- Check "Device" is set to GPU (not CPU)
- Reduce image size to 416 or 512
- Use smaller model (yolo11n)

### Issue: Dataset Validation Fails
**Solution:**
- Check data.yaml has correct paths
- Ensure images and labels match (same filenames)
- Verify label format (YOLO format: class x y w h)

### Issue: UI Freezes During Training
**Solution:** 
- This shouldn't happen! Training runs in separate thread
- If it does, please report as bug
- Try restarting application

### Issue: Cannot Import ultralytics
**Solution:**
```bash
pip install ultralytics --upgrade
```

### Issue: matplotlib Charts Not Showing
**Solution:**
```bash
pip install matplotlib
```

## Advanced Usage

### Custom Training Script
If you need more control, modify the `TrainingWorker` class in `yolo_trainer_gui.py`:

```python
class TrainingWorker(QThread):
    def run(self):
        # Add custom training logic here
        pass
```

### Add Custom Augmentations
In Tab 3, the augmentation parameters can be extended by modifying the `TrainingConfigTab` class.

### Export to Multiple Formats
```python
# In Results tab, you can programmatically export:
model.export(format='onnx')
model.export(format='engine', half=True)
model.export(format='tflite')
model.export(format='coreml')
```

## File Structure

```
ONNX_Training/
├── yolo_trainer_gui.py      # Main application
├── requirements.txt          # Dependencies
├── README.md                 # This file
└── runs/                     # Output folder (created automatically)
    └── train/
        └── exp_*/
            ├── weights/
            │   ├── best.pt  # Best model
            │   └── last.pt  # Last checkpoint
            ├── results.png  # Training curves
            └── ...
```

## Tips for Production Engineers

1. **Start Small:** Test with 10-20 images first to verify pipeline
2. **Monitor GPU:** Use `nvidia-smi` in terminal to check GPU usage
3. **Save Checkpoints:** Training auto-saves best.pt and last.pt
4. **Version Control:** Use experiment names with dates/versions
5. **Validate Often:** Run validation after training to check performance
6. **Export Early:** Test exports (ONNX/TensorRT) early in development
7. **Document:** Keep notes on what parameters work best for your use case

## Performance Optimization

### For Faster Training
- Increase batch size (if GPU memory allows)
- Use mixed precision (AMP checkbox - enabled by default)
- Reduce image size (416 instead of 640)
- Use more workers (8-16 if CPU has many cores)

### For Better Accuracy
- Train longer (more epochs)
- Increase image size (800 or 1024)
- Use larger model (yolo11m or yolo11l)
- Add more training data
- Increase data augmentation

### For Jetson Deployment
- Train with yolo11n or yolo11s
- Export to TensorRT with FP16
- Use image size 640 or smaller
- Test on Jetson before production

## Support and Resources

- **Ultralytics Docs:** https://docs.ultralytics.com
- **YOLO GitHub:** https://github.com/ultralytics/ultralytics
- **PySide6 Docs:** https://doc.qt.io/qtforpython/
- **TensorRT Docs:** https://docs.nvidia.com/deeplearning/tensorrt/

## License

This tool is provided as-is for production engineering purposes. Please refer to Ultralytics and PySide6 licenses for their respective components.

## Changelog

### Version 1.0
- Initial release
- Multi-tab interface
- Live training monitoring
- Threaded training
- Dataset validation
- Model export (ONNX, TensorRT, TFLite)
- Real-time charts
- File dialogs for all selections

---

**Built for Production Engineers - Made Easy** 🚀
