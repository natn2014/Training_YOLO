# Universal AI Inspection

A real-time YOLO object detection application with an intuitive GUI for AI-powered video inspection, industrial monitoring, and quality control analysis.

## Concept

Universal AI Inspection is a desktop application that leverages YOLOv8 deep learning models to perform real-time object detection on video streams from webcams or connected cameras. Built with PySide6, it provides a user-friendly interface for:

- Loading custom YOLO detection models
- Monitoring live video feeds with AI-powered object detection
- Filtering detections by confidence threshold and class type
- Setting up alerts for specific object classes
- Capturing and analyzing detection statistics
- Saving screenshots of detections for documentation

The application is designed for industrial inspection, quality control, surveillance, and research applications where real-time AI analysis of video streams is needed.

## Features

### Core Detection
- **Real-time YOLO Detection** - Live video analysis with YOLOv8 models
- **Multi-Camera Support** - Auto-detection of available cameras
- **Bounding Box Visualization** - Color-coded detection boxes with confidence scores

### Customization & Control
- **Custom Model Loading** - Load any YOLOv8 .pt model files
- **Confidence Filtering** - Adjustable threshold slider (0-100%)
- **Class Selection** - Toggle individual object classes on/off
- **Dynamic FPS Adjustment** - Auto-normalize camera FPS (24, 30, or 60 fps)

### Monitoring & Alerts
- **Match Alerts** - Real-time notifications when specific classes are detected
- **Detection Statistics** - Live table showing class counts per frame
- **Color-Coded Row Matching** - Visual correlation between detections and statistics

### Utilities
- **Frame Capture** - Screenshot detections to PNG, JPG, or BMP
- **Performance Optimization** - Automatic frame skipping for smooth real-time performance
- **Status Display** - Real-time monitoring of detection status and statistics

## Installation

### System Requirements

- **OS**: Windows 10+ or Linux/macOS with Python 3.8+
- **Python**: 3.8 or higher
- **GPU** (Optional): NVIDIA GPU with CUDA support for faster inference

### Step 1: Clone or Download the Project

```bash
# If using git
git clone <repository-url>
cd Universal_AI_Inspection

# Or extract the project folder
```

### Step 2: Create Virtual Environment (Recommended)

```bash
# Windows
python -m venv venv
venv\Scripts\activate

# Linux/macOS
python3 -m venv venv
source venv/bin/activate
```

### Step 3: Install Dependencies

```bash
pip install -r requirements.txt
```

**Dependencies:**
- `PySide6` - GUI framework
- `opencv-python` - Video processing
- `ultralytics` - YOLO implementation
- `numpy` - Numerical computing

### Step 4: Verify Installation

Test that all packages are properly installed:

```bash
python -c "import cv2, numpy, PySide6, ultralytics; print('All dependencies installed successfully!')"
```

### Step 5: Prepare YOLO Models

Download a YOLOv8 model file (.pt):

```bash
python -c "from ultralytics import YOLO; YOLO('yolov8n.pt')"
```

This will download and cache the nano model. You can also download other sizes:
- `yolov8n.pt` - Nano (smallest, fastest)
- `yolov8s.pt` - Small
- `yolov8m.pt` - Medium
- `yolov8l.pt` - Large
- `yolov8x.pt` - Extra Large (slowest, most accurate)

Alternatively, use custom-trained models in the same format.

### Step 6: Run the Application

```bash
python main.py
```

The GUI window should open. You're ready to start inspecting!

## Quick Start

1. **Load a Model** - Click "Load Model" and select a YOLOv8 .pt file
2. **Select Camera** - Choose your camera from the dropdown
3. **Click Start** - Begin real-time detection
4. **Adjust Filters** - Use the confidence slider and class checkboxes to refine results
5. **Set Alert Class** - Select a class to monitor for alerts
6. **Capture Frames** - Click "Capture Screenshot" to save detection images

## Troubleshooting

**No cameras detected:**
- Ensure your camera/webcam is properly connected
- Check camera permissions in Windows Settings
- Try restarting the application

**YOLO import error:**
- Verify ultralytics is installed: `pip install --upgrade ultralytics`
- Update CUDA drivers if using GPU acceleration

**Performance issues:**
- Use a smaller YOLO model (nano or small)
- Lower the FPS target
- Reduce video resolution if possible
- Ensure adequate GPU/CPU resources

## License and Credits

This project uses:
- [YOLOv8](https://github.com/ultralytics/ultralytics) - Ultralytics
- [PySide6](https://www.qt.io/) - Qt Framework
- [OpenCV](https://opencv.org/) - Computer Vision Library
