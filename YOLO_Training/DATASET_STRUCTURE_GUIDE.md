# YOLO Dataset Structure Guide

## Correct Directory Organization

### Option 1: Full Structure with Train/Val/Test Splits (Recommended)

Your dataset can be organized with explicit train/validation/test splits:

```
my_dataset/
├── train/
│   ├── images/
│   │   ├── image_001.jpg
│   │   ├── image_002.jpg
│   │   ├── image_003.jpg
│   │   └── ... (more training images)
│   └── labels/
│       ├── image_001.txt
│       ├── image_002.txt
│       ├── image_003.txt
│       └── ... (corresponding label files)
│
├── val/
│   ├── images/
│   │   ├── image_100.jpg
│   │   ├── image_101.jpg
│   │   └── ... (more validation images)
│   └── labels/
│       ├── image_100.txt
│       ├── image_101.txt
│       └── ... (corresponding label files)
│
├── test/ (OPTIONAL)
│   ├── images/
│   │   ├── image_200.jpg
│   │   ├── image_201.jpg
│   │   └── ... (more test images)
│   └── labels/
│       ├── image_200.txt
│       ├── image_201.txt
│       └── ... (corresponding label files)
│
└── data.yaml
```

### Option 2: Simple Structure (YOLO handles splits automatically)

For smaller datasets or when you want YOLO to automatically split your data:

```
my_dataset/
├── images/
│   ├── image_001.jpg
│   ├── image_002.jpg
│   ├── image_003.jpg
│   └── ... (more training images)
├── labels/
│   ├── image_001.txt
│   ├── image_002.txt
│   ├── image_003.txt
│   └── ... (corresponding label files)
└── data.yaml
```

**Note:** With this structure, YOLO can automatically split your data during training using the `split` parameter (e.g., 80% train, 20% val).

## Key Requirements

### 1. **Matching Image and Label Files**
   - **IMPORTANT:** Each image MUST have a corresponding label file
   - File names must match exactly (only extension differs):
     - ✅ CORRECT: `image_001.jpg` → `image_001.txt`
     - ❌ WRONG: `image_001.jpg` → `image_001.json`
     - ❌ WRONG: `image_001.jpg` → `001.txt`

### 2. **Train/Val/Test Split**
   - **train/**: 60-80% of your dataset (images + labels)
   - **val/**: 10-20% of your dataset (images + labels)
   - **test/**: 10-20% of your dataset (optional, images + labels)

### 3. **Label Format (YOLO Format)**
Each `.txt` label file should contain:
```
<class_id> <x_center> <y_center> <width> <height>
<class_id> <x_center> <y_center> <width> <height>
...
```

**Important Notes:**
- One line per object in the image
- All coordinates are **normalized to 0-1** range
- `class_id` starts from 0
- Format: space-separated values

**Example label file (`image_001.txt`):**
```
0 0.5 0.5 0.3 0.4
1 0.25 0.25 0.15 0.2
```
This means:
- Object of class 0 (person) centered at (0.5, 0.5) with width 0.3 and height 0.4
- Object of class 1 (car) centered at (0.25, 0.25) with width 0.15 and height 0.2

### 4. **data.yaml Configuration**
```yaml
path: /absolute/path/to/my_dataset  # Absolute path to dataset root

# DO NOT CHANGE THESE - they match the directory structure
train: train
val: val
test: test  # optional

nc: 3  # Number of classes

names:
  0: person
  1: car
  2: bicycle
  # ... add all your classes
```

## Common Mistakes

### ❌ WRONG Structure 1: Flat Images/Labels
```
my_dataset/
├── images/
│   ├── train/
│   ├── val/
│   └── test/
└── labels/
    ├── train/
    ├── val/
    └── test/
```
**Problem:** The training code expects images and labels together in each split folder.

### ❌ WRONG Structure 2: Only Images, No Labels
```
my_dataset/
├── train/
│   └── images/  (only images, no labels!)
├── val/
│   └── images/
└── data.yaml
```
**Problem:** Each image needs a corresponding label file.

### ❌ WRONG Structure 3: Mismatched File Names
```
train/images/
├── image_1.jpg
├── photo.jpg
└── pic.jpg

train/labels/
├── image_1.txt
├── img_photo.txt     ← MISMATCH! Should be "photo.txt"
└── picture.txt       ← MISMATCH! Should be "pic.txt"
```
**Problem:** Label files must match image file names exactly.

## Correct Structure Examples

### Example 1: Pet Detection Dataset
```
pet_dataset/
├── train/
│   ├── images/
│   │   ├── cat_001.jpg
│   │   ├── cat_002.jpg
│   │   ├── dog_001.jpg
│   │   └── dog_002.jpg
│   └── labels/
│       ├── cat_001.txt
│       ├── cat_002.txt
│       ├── dog_001.txt
│       └── dog_002.txt
├── val/
│   ├── images/
│   │   ├── cat_003.jpg
│   │   └── dog_003.jpg
│   └── labels/
│       ├── cat_003.txt
│       └── dog_003.txt
└── data.yaml
```

### Example 2: Industrial Defect Detection
```
defect_dataset/
├── train/
│   ├── images/
│   │   ├── product_0001.jpg
│   │   ├── product_0002.jpg
│   │   └── ...
│   └── labels/
│       ├── product_0001.txt
│       ├── product_0002.txt
│       └── ...
├── val/
│   ├── images/
│   │   ├── product_0500.jpg
│   │   └── ...
│   └── labels/
│       ├── product_0500.txt
│       └── ...
└── data.yaml
```

## Validation with GUI

The YOLO Training GUI will:
1. Count all images in `train/images/`, `val/images/`, `test/images/`
2. Count all labels in `train/labels/`, `val/labels/`, `test/labels/`
3. **Alert you if counts don't match** (missing labels or extra images)
4. Show statistics table with:
   - Train/Val/Test image counts
   - Train/Val/Test label counts
   - Total images
   - Dataset path

### What the Validation Shows

```
Property                 Value
─────────────────────────────────────────
Number of Classes        3
Class Names              person, car, bicycle
Train Images            800
Train Labels            800     ✓ Matches!
Validation Images       200
Validation Labels       200     ✓ Matches!
Test Images              0
Test Labels              0      ✓ OK (Optional)
Total Images            1000
Dataset Path            D:\datasets\my_dataset
```

If counts don't match, you'll see a warning:
```
Dataset validation found issues:
Train: 800 images but 795 labels
Make sure all images have corresponding label files.
```

## Creating Labels

### Option 1: LabelImg (GUI Tool)
```bash
pip install labelImg
labelImg
```
- Export in "YOLO" format
- Automatically creates `.txt` files

### Option 2: Roboflow (Online)
- Visit https://roboflow.com
- Upload images
- Annotate online
- Export in YOLO format
- Download with correct structure

### Option 3: Label Studio (Web Tool)
Already running at: http://localhost:8080
- Can export to YOLO format
- Make sure to organize output correctly

### Option 4: Manual Creation
Create `.txt` files programmatically:
```python
import cv2
from pathlib import Path

# For each image, create a label file
image = cv2.imread('train/images/image_001.jpg')
height, width = image.shape[:2]

# If object detected at x=100, y=150, w=50, h=70 of class 0:
x_center = (100 + 50/2) / width  # normalize
y_center = (150 + 70/2) / height  # normalize
w_norm = 50 / width
h_norm = 70 / height

with open('train/labels/image_001.txt', 'w') as f:
    f.write(f"0 {x_center} {y_center} {w_norm} {h_norm}\n")
```

## Troubleshooting

### Issue: "Train Images: 100, Train Labels: 95"
**Solution:** Find missing labels
```bash
# In train/labels folder, check which images don't have labels:
# If images/image_001.jpg exists but labels/image_001.txt doesn't, create the label file
```

### Issue: File name mismatches
**Solution:** Rename label files to match images
```bash
# Wrong:
train/images/photo.jpg
train/labels/img_photo.txt

# Correct:
train/images/photo.jpg
train/labels/photo.txt
```

### Issue: Validation passes but training fails
**Solution:** Check label format
- Verify `.txt` files contain valid YOLO format
- All coordinates should be between 0 and 1
- Check for special characters or empty lines

## Verification Script

Run this Python script to verify your dataset structure:

```python
from pathlib import Path
import sys

def verify_dataset(dataset_root):
    """Verify dataset structure is correct"""
    
    issues = []
    
    for split in ['train', 'val', 'test']:
        split_path = Path(dataset_root) / split
        if not split_path.exists():
            continue
            
        images_path = split_path / 'images'
        labels_path = split_path / 'labels'
        
        if not images_path.exists() or not labels_path.exists():
            issues.append(f"{split}: Missing images or labels folder")
            continue
        
        images = set(p.stem for p in images_path.glob('*'))
        labels = set(p.stem for p in labels_path.glob('*'))
        
        missing_labels = images - labels
        extra_labels = labels - images
        
        if missing_labels:
            issues.append(f"{split}: Missing labels for {missing_labels}")
        if extra_labels:
            issues.append(f"{split}: Extra labels {extra_labels}")
    
    if issues:
        print("❌ Issues found:")
        for issue in issues:
            print(f"  - {issue}")
        return False
    else:
        print("✅ Dataset structure is correct!")
        return True

# Usage:
if __name__ == "__main__":
    verify_dataset("D:/my_dataset")
```

## Summary

✅ **Required Structure:**
```
dataset_root/
├── train/ → [images/ + labels/]
├── val/   → [images/ + labels/]
├── test/  → [images/ + labels/] (optional)
└── data.yaml
```

✅ **Requirements:**
- Image and label files must have matching names
- Labels must be in YOLO format (class_id x_center y_center width height)
- Coordinates must be normalized (0-1)
- data.yaml must point to correct paths

✅ **Validation:**
The GUI will check all this for you and alert if something is wrong!

