"""
YOLO Training GUI Application
Target User: Production Engineers
Features: Multi-tab interface, file dialogs, live charts, threaded training
"""

import sys
import os
import subprocess
import platform
from pathlib import Path
from PySide6.QtWidgets import (
    QApplication, QMainWindow, QTabWidget, QWidget, QVBoxLayout,
    QHBoxLayout, QPushButton, QLabel, QLineEdit, QComboBox,
    QSpinBox, QDoubleSpinBox, QTextEdit, QProgressBar, QFileDialog,
    QGroupBox, QGridLayout, QTableWidget, QTableWidgetItem, QMessageBox,
    QSplitter, QCheckBox, QScrollArea, QInputDialog
)
from PySide6.QtCore import Qt, QThread, Signal, Slot
from PySide6.QtGui import QFont, QTextCursor, QPixmap, QIcon
import yaml
from typing import Dict, List, Optional
from datetime import datetime

# Import theme manager
from ui_styles import ThemeManager

# Import for charts
try:
    from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
    from matplotlib.figure import Figure
    import matplotlib.pyplot as plt
    MATPLOTLIB_AVAILABLE = True
except ImportError:
    MATPLOTLIB_AVAILABLE = False
    print("Warning: matplotlib not available. Charts will be disabled.")


class TrainingWorker(QThread):
    """Worker thread for YOLO training to prevent UI freezing"""
    
    # Signals for communication with main thread
    progress_update = Signal(int)  # Progress percentage
    epoch_update = Signal(dict)  # Epoch metrics
    status_update = Signal(str)  # Status messages
    training_complete = Signal(str)  # Path to best.pt
    training_error = Signal(str)  # Error message
    
    def __init__(self, config: Dict):
        super().__init__()
        self.config = config
        self.is_running = True
        
    def run(self):
        """Execute training in separate thread"""
        try:
            from ultralytics import YOLO  # type: ignore
            from pathlib import Path
            import os
            
            # Check ultralytics version for YOLOv11 support
            try:
                import ultralytics
                version = ultralytics.__version__
                major, minor = map(int, version.split('.')[:2])
                
                model_path = self.config['pretrained_model']
                
                # Check if YOLOv11 is being used with old ultralytics version
                if 'yolo11' in model_path.lower() or 'v11' in model_path.lower():
                    if major < 8 or (major == 8 and minor < 1):
                        self.status_update.emit(f"Warning: YOLOv11 requires ultralytics>=8.1.0 (current: {version})")
                        self.status_update.emit("Falling back to YOLOv8n model...")
                        model_path = model_path.replace('yolo11', 'yolov8').replace('11', '8')
                        self.status_update.emit(f"Using {model_path} instead")
            except Exception:
                pass
            
            model_path = self.config['pretrained_model']
            
            # Check if it's a pre-trained model name (not a full path)
            if not os.path.isabs(model_path):
                # Check if file exists in current directory or ultralytics cache
                if not Path(model_path).exists():
                    # It's a model name that will be auto-downloaded by ultralytics
                    self.status_update.emit(f"Downloading pre-trained model {model_path}...")
                    self.status_update.emit("This may take a few minutes on first use...")
                    
                    # Let ultralytics handle the download by just passing the model name
                    try:
                        model = YOLO(model_path)
                    except AttributeError as e:
                        if 'C3k2' in str(e) or 'C2PSA' in str(e):
                            # YOLOv11 model with old ultralytics version
                            self.training_error.emit(
                                "YOLOv11 models require ultralytics>=8.1.0\n\n"
                                "Please upgrade ultralytics:\n"
                                "pip install --upgrade ultralytics\n\n"
                                "Or select YOLOv8 models instead (yolov8n.pt, yolov8s.pt, etc.)"
                            )
                            return
                        else:
                            raise
                    except Exception as e:
                        # If it fails, try without .pt extension (ultralytics sometimes prefers this)
                        model_name_no_ext = model_path.replace('.pt', '')
                        self.status_update.emit(f"Retrying with model name: {model_name_no_ext}")
                        try:
                            model = YOLO(model_name_no_ext)
                        except AttributeError as e:
                            if 'C3k2' in str(e) or 'C2PSA' in str(e):
                                self.training_error.emit(
                                    "YOLOv11 models require ultralytics>=8.1.0\n\n"
                                    "Please upgrade ultralytics:\n"
                                    "pip install --upgrade ultralytics\n\n"
                                    "Or select YOLOv8 models instead (yolov8n.pt, yolov8s.pt, etc.)"
                                )
                                return
                            else:
                                raise
                else:
                    self.status_update.emit("Loading pre-trained model...")
                    model = YOLO(model_path)
            else:
                # It's an absolute path to a custom model
                self.status_update.emit("Loading custom model...")
                model = YOLO(model_path)
            
            self.status_update.emit("Starting training...")
            
            # Custom callback for progress updates
            def on_train_epoch_end(trainer):
                """Called at the end of each training epoch"""
                if not self.is_running:
                    trainer.stop = True
                    return
                    
                try:
                    epoch = trainer.epoch + 1  # trainer.epoch is 0-indexed
                    epochs = trainer.epochs
                    progress = int((epoch / epochs) * 100)
                    
                    # Get metrics from trainer
                    metrics = {
                        'epoch': epoch,
                        'epochs': epochs,
                        'box_loss': 0.0,
                        'cls_loss': 0.0,
                        'dfl_loss': 0.0,
                        'precision': 0.0,
                        'recall': 0.0,
                        'mAP50': 0.0,
                        'mAP50-95': 0.0,
                    }
                    
                    # Extract loss values
                    if hasattr(trainer, 'loss_items') and trainer.loss_items is not None:
                        loss_items = trainer.loss_items
                        if len(loss_items) >= 3:
                            metrics['box_loss'] = float(loss_items[0])
                            metrics['cls_loss'] = float(loss_items[1])
                            metrics['dfl_loss'] = float(loss_items[2])
                    
                    # Extract validation metrics if available
                    if hasattr(trainer, 'metrics') and trainer.metrics is not None:
                        metrics['precision'] = float(trainer.metrics.get('metrics/precision(B)', 0))
                        metrics['recall'] = float(trainer.metrics.get('metrics/recall(B)', 0))
                        metrics['mAP50'] = float(trainer.metrics.get('metrics/mAP50(B)', 0))
                        metrics['mAP50-95'] = float(trainer.metrics.get('metrics/mAP50-95(B)', 0))
                    
                    self.progress_update.emit(progress)
                    self.epoch_update.emit(metrics)
                except Exception as e:
                    self.status_update.emit(f"Callback error: {str(e)}")
            
            # Register callback with model
            model.add_callback("on_train_epoch_end", on_train_epoch_end)
            
            # Train with callbacks
            results = model.train(
                data=self.config['data_yaml'],
                epochs=self.config['epochs'],
                imgsz=self.config['imgsz'],
                batch=self.config['batch'],
                device=self.config['device'],
                project=self.config['project'],
                name=self.config['name'],
                patience=self.config['patience'],
                optimizer=self.config['optimizer'],
                lr0=self.config['lr0'],
                verbose=True,
                plots=True,
                val=True,
                amp=self.config.get('amp', True),
                workers=self.config.get('workers', 8),
            )
            
            if results and hasattr(results, 'save_dir') and results.save_dir:
                best_model_path = str(Path(results.save_dir) / 'weights' / 'best.pt')
                self.training_complete.emit(best_model_path)
            else:
                self.training_error.emit("Training completed but could not locate best model.")
            
        except Exception as e:
            self.training_error.emit(str(e))
            
    def stop(self):
        """Stop training"""
        self.is_running = False


class DatasetTab(QWidget):
    """Tab 1: Dataset Selection and Validation"""
    
    def __init__(self, theme_manager):
        super().__init__()
        self.theme_manager = theme_manager
        self.dataset_path = None
        self.data_yaml_path = None
        self.dataset_info = {}
        self.init_ui()
        
    def init_ui(self):
        layout = QVBoxLayout()
        
        # Dataset Selection Group
        dataset_group = QGroupBox("Dataset Selection")
        dataset_group.setStyleSheet(self.theme_manager.get_theme().get_groupbox_stylesheet())
        dataset_layout = QVBoxLayout()
        
        # Dataset path selection
        path_layout = QHBoxLayout()
        path_layout.addWidget(QLabel("Dataset Folder:"))
        self.dataset_path_input = QLineEdit()
        self.dataset_path_input.setPlaceholderText("Select dataset root folder...")
        self.dataset_path_input.setReadOnly(True)
        path_layout.addWidget(self.dataset_path_input)
        
        browse_btn = QPushButton("Browse...")
        browse_btn.setStyleSheet(self.theme_manager.get_theme().get_button_stylesheet())
        browse_btn.clicked.connect(self.browse_dataset)
        path_layout.addWidget(browse_btn)
        dataset_layout.addLayout(path_layout)
        
        # Data.yaml selection
        yaml_layout = QHBoxLayout()
        yaml_layout.addWidget(QLabel("data.yaml File:"))
        self.yaml_path_input = QLineEdit()
        self.yaml_path_input.setPlaceholderText("Select data.yaml file...")
        self.yaml_path_input.setReadOnly(True)
        yaml_layout.addWidget(self.yaml_path_input)
        
        browse_yaml_btn = QPushButton("Browse...")
        browse_yaml_btn.setStyleSheet(self.theme_manager.get_theme().get_button_stylesheet())
        browse_yaml_btn.clicked.connect(self.browse_yaml)
        yaml_layout.addWidget(browse_yaml_btn)
        dataset_layout.addLayout(yaml_layout)
        
        # Validate button
        validate_btn = QPushButton("Validate Dataset")
        validate_btn.setStyleSheet(self.theme_manager.get_theme().get_button_stylesheet('primary'))
        validate_btn.clicked.connect(self.validate_dataset)
        dataset_layout.addWidget(validate_btn)
        
        dataset_group.setLayout(dataset_layout)
        layout.addWidget(dataset_group)
        
        # Dataset Statistics Group
        stats_group = QGroupBox("Dataset Statistics")
        stats_group.setStyleSheet(self.theme_manager.get_theme().get_groupbox_stylesheet())
        stats_layout = QGridLayout()
        
        self.stats_table = QTableWidget()
        self.stats_table.setStyleSheet(self.theme_manager.get_theme().get_table_stylesheet())
        self.stats_table.setColumnCount(2)
        self.stats_table.setHorizontalHeaderLabels(["Property", "Value"])
        self.stats_table.horizontalHeader().setStretchLastSection(True)
        stats_layout.addWidget(self.stats_table, 0, 0, 1, 2)
        
        stats_group.setLayout(stats_layout)
        layout.addWidget(stats_group)
        
        # Chart for data distribution
        if MATPLOTLIB_AVAILABLE:
            chart_group = QGroupBox("Data Distribution")
            chart_group.setStyleSheet(self.theme_manager.get_theme().get_groupbox_stylesheet())
            chart_layout = QVBoxLayout()
            
            self.figure = Figure(figsize=(10, 4))
            self.canvas = FigureCanvas(self.figure)
            chart_layout.addWidget(self.canvas)
            
            chart_group.setLayout(chart_layout)
            layout.addWidget(chart_group)
        
        layout.addStretch()
        self.setLayout(layout)
        
    def browse_dataset(self):
        """Browse for dataset folder"""
        folder = QFileDialog.getExistingDirectory(self, "Select Dataset Folder")
        if folder:
            self.dataset_path = folder
            self.dataset_path_input.setText(folder)
            
            # Auto-detect data.yaml
            yaml_path = Path(folder) / 'data.yaml'
            if yaml_path.exists():
                self.data_yaml_path = str(yaml_path)
                self.yaml_path_input.setText(str(yaml_path))
                
    def browse_yaml(self):
        """Browse for data.yaml file"""
        file, _ = QFileDialog.getOpenFileName(
            self, "Select data.yaml", "", "YAML Files (*.yaml *.yml)"
        )
        if file:
            self.data_yaml_path = file
            self.yaml_path_input.setText(file)
    
    def create_simple_structure_yaml(self, dataset_root):
        """Create data.yaml for simple structure dataset (images/ and labels/ at root)"""
        try:
            images_folder = dataset_root / 'images'
            labels_folder = dataset_root / 'labels'
            
            # Count images and labels
            image_files = list(images_folder.glob('*.[jJ][pP][gG]')) + \
                         list(images_folder.glob('*.[jJ][pP][eE][gG]')) + \
                         list(images_folder.glob('*.[pP][nN][gG]')) + \
                         list(images_folder.glob('*.[bB][mM][pP]')) + \
                         list(images_folder.glob('*.[wW][eE][bB][pP]'))
            
            label_files = list(labels_folder.glob('*.txt'))
            
            total_images = len(image_files)
            total_labels = len(label_files)
            
            if total_images == 0:
                QMessageBox.warning(self, "Error", "No images found in images/ folder!")
                return False
            
            if total_labels == 0:
                QMessageBox.warning(self, "Error", "No label files found in labels/ folder!")
                return False
            
            if total_images != total_labels:
                reply = QMessageBox.question(
                    self, 'Warning',
                    f'Image count ({total_images}) does not match label count ({total_labels}).\n\n'
                    'Some images may not have labels or vice versa.\n'
                    'Continue anyway?',
                    QMessageBox.Yes | QMessageBox.No,
                    QMessageBox.No
                )
                if reply == QMessageBox.No:
                    return False
            
            # Try to detect class names from label files
            class_ids = set()
            for label_file in label_files[:100]:  # Sample first 100 files
                try:
                    with open(label_file, 'r') as f:
                        for line in f:
                            parts = line.strip().split()
                            if parts:
                                class_ids.add(int(parts[0]))
                except:
                    continue
            
            num_classes = len(class_ids) if class_ids else 1
            class_names = [f'class_{i}' for i in range(num_classes)]
            
            # Try to read class names from classes.txt if it exists
            classes_txt_path = dataset_root / 'classes.txt'
            if classes_txt_path.exists():
                try:
                    with open(classes_txt_path, 'r') as f:
                        file_class_names = [line.strip() for line in f.readlines() if line.strip()]
                    if file_class_names:
                        class_names = file_class_names
                        num_classes = len(class_names)
                except:
                    pass
            
            # Ask user to confirm or edit class names
            class_names_str = ', '.join(class_names)
            class_names_input, ok = QInputDialog.getText(
                self, 'Class Names',
                f'Detected {num_classes} classes.\n\n'
                f'Enter class names (comma-separated):',
                text=class_names_str
            )
            
            if ok and class_names_input.strip():
                class_names = [name.strip() for name in class_names_input.split(',')]
                num_classes = len(class_names)
            elif not ok:
                return False
            
            # Create data.yaml content
            yaml_content = {
                'path': str(dataset_root.absolute()),
                'train': 'images',  # YOLO will automatically split
                'val': 'images',    # YOLO will automatically split
                'nc': num_classes,
                'names': class_names
            }
            
            # Add comment about split
            yaml_path = dataset_root / 'data.yaml'
            with open(yaml_path, 'w') as f:
                f.write('# Auto-generated data.yaml for simple structure dataset\n')
                f.write('# Dataset will be automatically split 80% train / 20% val during training\n')
                f.write('# Use split parameter in training: model.train(data="data.yaml", split=0.8)\n\n')
                yaml.dump(yaml_content, f, default_flow_style=False, sort_keys=False)
            
            self.data_yaml_path = str(yaml_path)
            self.yaml_path_input.setText(str(yaml_path))
            
            QMessageBox.information(
                self, 'Success',
                f'data.yaml created successfully!\n\n'
                f'Location: {yaml_path}\n'
                f'Total images: {total_images}\n'
                f'Total labels: {total_labels}\n'
                f'Classes: {num_classes}\n\n'
                f'Note: Dataset will be split 80% train / 20% val during training.'
            )
            
            return True
            
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Failed to create data.yaml:\n{str(e)}")
            return False
            
    def validate_dataset(self):
        """Validate dataset structure and load statistics"""
        # Check if we need to auto-create data.yaml for simple structure
        if not self.data_yaml_path and self.dataset_path:
            # Check if dataset uses simple structure (images/ and labels/ at root)
            dataset_root = Path(self.dataset_path)
            images_folder = dataset_root / 'images'
            labels_folder = dataset_root / 'labels'
            
            if images_folder.exists() and labels_folder.exists():
                # Simple structure detected - offer to create data.yaml
                reply = QMessageBox.question(
                    self, 'Simple Dataset Structure Detected',
                    'This dataset uses the simple structure (images/ and labels/ at root).\n\n'
                    'Would you like to automatically create a data.yaml file\n'
                    'with 80% train / 20% validation split?',
                    QMessageBox.Yes | QMessageBox.No,
                    QMessageBox.Yes
                )
                
                if reply == QMessageBox.Yes:
                    if not self.create_simple_structure_yaml(dataset_root):
                        return
                else:
                    QMessageBox.warning(self, "Warning", "Please select or create a data.yaml file first!")
                    return
            else:
                QMessageBox.warning(self, "Warning", "Please select data.yaml file first!")
                return
        
        if not self.data_yaml_path:
            QMessageBox.warning(self, "Warning", "Please select data.yaml file first!")
            return
            
        try:
            # Load data.yaml
            with open(self.data_yaml_path, 'r') as f:
                data_config = yaml.safe_load(f)
                
            self.dataset_info = data_config
            
            # Count images with correct directory structure
            # Expected structure:
            # dataset_root/
            # ├── train/
            # │   ├── images/
            # │   └── labels/
            # ├── val/ (or valid/)
            # │   ├── images/
            # │   └── labels/
            # └── test/ (optional)
            #     ├── images/
            #     └── labels/
            
            dataset_root = Path(data_config.get('path', Path(self.data_yaml_path).parent))
            
            # Check if this is a simple structure (images/ and labels/ at root)
            is_simple_structure = False
            simple_images_path = dataset_root / 'images'
            simple_labels_path = dataset_root / 'labels'
            
            if simple_images_path.exists() and simple_labels_path.exists():
                is_simple_structure = True
                # For simple structure, count images at root level
                image_files_list = [f for f in simple_images_path.glob('*') if f.suffix.lower() in ['.jpg', '.jpeg', '.png', '.bmp', '.webp']]
                train_count = len(image_files_list)
                
                # Check for matching label files
                train_labels_count = 0
                for img_file in image_files_list:
                    label_file = simple_labels_path / f'{img_file.stem}.txt'
                    if label_file.exists():
                        train_labels_count += 1
                
                val_count = 0
                val_labels_count = 0
                test_count = 0
                test_labels_count = 0
            else:
                # Count images in train/images folder
                train_images_path = dataset_root / 'train' / 'images'
                train_labels_path = dataset_root / 'train' / 'labels'
                train_image_files = list(train_images_path.glob('*')) if train_images_path.exists() else []
                train_count = len(train_image_files)
                
                # Check for matching label files in train
                train_labels_count = 0
                for img_file in train_image_files:
                    label_file = train_labels_path / f'{img_file.stem}.txt'
                    if label_file.exists():
                        train_labels_count += 1
                
                # Count images in val/images folder OR valid/images folder
                val_images_path = dataset_root / 'val' / 'images'
                val_labels_path = dataset_root / 'val' / 'labels'
                
                # Check for 'valid' as alternative
                if not val_images_path.exists():
                    val_images_path = dataset_root / 'valid' / 'images'
                    val_labels_path = dataset_root / 'valid' / 'labels'
                
                val_image_files = list(val_images_path.glob('*')) if val_images_path.exists() else []
                val_count = len(val_image_files)
                
                # Check for matching label files in val
                val_labels_count = 0
                for img_file in val_image_files:
                    label_file = val_labels_path / f'{img_file.stem}.txt'
                    if label_file.exists():
                        val_labels_count += 1
                
                # Count images in test/images folder (optional)
                test_images_path = dataset_root / 'test' / 'images'
                test_labels_path = dataset_root / 'test' / 'labels'
                test_image_files = list(test_images_path.glob('*')) if test_images_path.exists() else []
                test_count = len(test_image_files)
                
                # Check for matching label files in test
                test_labels_count = 0
                for img_file in test_image_files:
                    label_file = test_labels_path / f'{img_file.stem}.txt'
                    if label_file.exists():
                        test_labels_count += 1
            
            # Validate that images and labels match
            validation_issues = []
            if is_simple_structure:
                if train_count > 0 and train_count != train_labels_count:
                    validation_issues.append(f"Images: {train_count} but Labels: {train_labels_count}")
                # Calculate estimated split
                estimated_train = int(train_count * 0.8)
                estimated_val = train_count - estimated_train
            else:
                if train_count > 0 and train_count != train_labels_count:
                    validation_issues.append(f"Train: {train_count} images but {train_labels_count} labels")
                if val_count > 0 and val_count != val_labels_count:
                    validation_issues.append(f"Val: {val_count} images but {val_labels_count} labels")
                if test_count > 0 and test_count != test_labels_count:
                    validation_issues.append(f"Test: {test_count} images but {test_labels_count} labels")
            
            # Update statistics table
            self.stats_table.setRowCount(0)
            
            if is_simple_structure:
                estimated_train = int(train_count * 0.8)
                estimated_val = train_count - estimated_train
                stats = [
                    ("Dataset Structure", "Simple (auto-split)"),
                    ("Number of Classes", str(data_config.get('nc', 0))),
                    ("Class Names", ", ".join(data_config.get('names', {}).values() if isinstance(data_config.get('names'), dict) else data_config.get('names', []))),
                    ("Total Images", str(train_count)),
                    ("Total Labels", str(train_labels_count)),
                    ("Estimated Train (80%)", str(estimated_train)),
                    ("Estimated Val (20%)", str(estimated_val)),
                    ("Test Images", "0 (N/A)"),
                    ("Dataset Path", str(dataset_root)),
                ]
            else:
                stats = [
                    ("Dataset Structure", "Full (train/val/test)"),
                    ("Number of Classes", str(data_config.get('nc', 0))),
                    ("Class Names", ", ".join(data_config.get('names', {}).values() if isinstance(data_config.get('names'), dict) else data_config.get('names', []))),
                    ("Train Images", str(train_count)),
                    ("Train Labels", str(train_labels_count)),
                    ("Validation Images", str(val_count)),
                    ("Validation Labels", str(val_labels_count)),
                    ("Test Images", str(test_count)),
                    ("Test Labels", str(test_labels_count)),
                    ("Total Images", str(train_count + val_count + test_count)),
                    ("Dataset Path", str(dataset_root)),
                ]
            
            for prop, value in stats:
                row = self.stats_table.rowCount()
                self.stats_table.insertRow(row)
                self.stats_table.setItem(row, 0, QTableWidgetItem(prop))
                self.stats_table.setItem(row, 1, QTableWidgetItem(value))
            
            # Show warning if images/labels don't match
            if validation_issues:
                warning_msg = "Dataset validation found issues:\n\n" + "\n".join(validation_issues) + "\n\nMake sure all images have corresponding label files."
                QMessageBox.warning(self, "Validation Warning", warning_msg)
            
            # Update chart
            if MATPLOTLIB_AVAILABLE:
                if is_simple_structure:
                    estimated_train = int(train_count * 0.8)
                    estimated_val = train_count - estimated_train
                    self.update_chart(estimated_train, estimated_val, 0, data_config.get('names', {}))
                else:
                    self.update_chart(train_count, val_count, test_count, data_config.get('names', {}))
            
            # Success message
            if not validation_issues:
                QMessageBox.information(self, "Success", "Dataset validated successfully!")
            
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Failed to validate dataset:\n{str(e)}")
            
    def update_chart(self, train_count, val_count, test_count, class_names):
        """Update data distribution chart"""
        self.figure.clear()
        
        # Create two subplots
        ax1 = self.figure.add_subplot(121)
        ax2 = self.figure.add_subplot(122)
        
        # Plot 1: Split distribution
        splits = ['Train', 'Validation', 'Test']
        counts = [train_count, val_count, test_count]
        colors = ['#2ecc71', '#3498db', '#e74c3c']
        
        ax1.bar(splits, counts, color=colors)
        ax1.set_ylabel('Number of Images')
        ax1.set_title('Dataset Split Distribution')
        ax1.grid(axis='y', alpha=0.3)
        
        # Add value labels on bars
        for i, (split, count) in enumerate(zip(splits, counts)):
            ax1.text(i, count, str(count), ha='center', va='bottom')
            
        # Plot 2: Class distribution (pie chart with actual counts from labels)
        if isinstance(class_names, dict):
            class_list = list(class_names.values())
            num_classes = len(class_names)
        else:
            class_list = class_names
            num_classes = len(class_list)
        
        # Count actual class occurrences from label files
        try:
            if not self.data_yaml_path:
                raise ValueError("data_yaml_path is not set")
            dataset_root = Path(self.data_yaml_path).parent
            if hasattr(self, 'dataset_info') and 'path' in self.dataset_info:
                dataset_root = Path(self.dataset_info['path'])
            
            class_counts = [0] * num_classes
            
            # Get all label directories
            label_dirs = []
            
            # First, check for simple structure (labels at root)
            simple_labels = dataset_root / 'labels'
            if simple_labels.exists():
                label_dirs.append(simple_labels)
            
            # Then check for full structure (train/val/test)
            train_labels = dataset_root / 'train' / 'labels'
            if train_labels.exists():
                label_dirs.append(train_labels)
            
            val_labels = dataset_root / 'val' / 'labels'
            if val_labels.exists():
                label_dirs.append(val_labels)
            else:
                valid_labels = dataset_root / 'valid' / 'labels'
                if valid_labels.exists():
                    label_dirs.append(valid_labels)
            
            test_labels = dataset_root / 'test' / 'labels'
            if test_labels.exists():
                label_dirs.append(test_labels)
            
            # Count class occurrences in all label files
            for label_dir in label_dirs:
                for label_file in label_dir.glob('*.txt'):
                    try:
                        with open(label_file, 'r') as f:
                            for line in f:
                                parts = line.strip().split()
                                if parts:
                                    class_id = int(parts[0])
                                    if 0 <= class_id < num_classes:
                                        class_counts[class_id] += 1
                    except Exception:
                        continue
            
            total_objects = sum(class_counts)
            
            # Create pie chart with actual counts
            if total_objects > 0 and class_list:
                # Filter out classes with zero counts for cleaner visualization
                non_zero_classes = [(class_list[i], class_counts[i]) for i in range(len(class_list)) if class_counts[i] > 0]
                
                if non_zero_classes and len(non_zero_classes) <= 15:  # Show pie if reasonable number
                    labels_with_counts = [f'{name}\n({count})' for name, count in non_zero_classes]
                    values = [count for _, count in non_zero_classes]
                    
                    ax2.pie(values, labels=labels_with_counts, autopct='%1.1f%%', startangle=90)
                    ax2.set_title(f'Class Distribution ({total_objects} total objects)')
                else:
                    # Too many classes, show summary
                    summary_text = f'{len(non_zero_classes)} Classes\n{total_objects} Total Objects'
                    ax2.text(0.5, 0.5, summary_text, ha='center', va='center', fontsize=14)
                    ax2.set_xlim(0, 1)
                    ax2.set_ylim(0, 1)
                    ax2.axis('off')
                    ax2.set_title('Class Distribution')
            else:
                ax2.text(0.5, 0.5, f'{len(class_list)} Classes\nNo labels found', 
                        ha='center', va='center', fontsize=14)
                ax2.set_xlim(0, 1)
                ax2.set_ylim(0, 1)
                ax2.axis('off')
                ax2.set_title('Class Distribution')
                
        except Exception as e:
            # Fallback to simple display if counting fails
            ax2.text(0.5, 0.5, f'{len(class_list)} Classes\n(Count failed)', 
                    ha='center', va='center', fontsize=14)
            ax2.set_xlim(0, 1)
            ax2.set_ylim(0, 1)
            ax2.axis('off')
            ax2.set_title('Class Distribution')
            
        self.figure.tight_layout()
        self.canvas.draw()


class ModelSelectionTab(QWidget):
    """Tab 2: Pre-trained Model Selection"""
    
    def __init__(self, theme_manager):
        super().__init__()
        self.theme_manager = theme_manager
        self.selected_model = None
        self.init_ui()
        
    def init_ui(self):
        layout = QVBoxLayout()
        
        # Model Selection Group
        model_group = QGroupBox("Pre-trained Model Selection")
        model_group.setStyleSheet(self.theme_manager.get_theme().get_groupbox_stylesheet())
        model_layout = QVBoxLayout()
        
        # Model type selection
        type_layout = QHBoxLayout()
        type_layout.addWidget(QLabel("Model Source:"))
        self.model_source_combo = QComboBox()
        self.model_source_combo.addItems([
            "YOLOv11 Pre-trained",
            "YOLOv8 Pre-trained",
            "YOLOv10 Pre-trained",
            "Custom .PT File"
        ])
        self.model_source_combo.currentTextChanged.connect(self.on_source_changed)
        type_layout.addWidget(self.model_source_combo)
        type_layout.addStretch()
        model_layout.addLayout(type_layout)
        
        # Pre-trained model dropdown
        pretrained_layout = QHBoxLayout()
        pretrained_layout.addWidget(QLabel("Select Model:"))
        self.pretrained_combo = QComboBox()
        self.pretrained_combo.addItems([
            "yolo11n.pt (Nano - Fastest)",
            "yolo11s.pt (Small - Balanced)",
            "yolo11m.pt (Medium - Accurate)",
            "yolo11l.pt (Large - More Accurate)",
            "yolo11x.pt (XLarge - Most Accurate)"
        ])
        pretrained_layout.addWidget(self.pretrained_combo)
        pretrained_layout.addStretch()
        model_layout.addLayout(pretrained_layout)
        
        # Custom model file selection
        custom_layout = QHBoxLayout()
        custom_layout.addWidget(QLabel("Custom Model:"))
        self.custom_model_input = QLineEdit()
        self.custom_model_input.setPlaceholderText("Select custom .pt file...")
        self.custom_model_input.setReadOnly(True)
        self.custom_model_input.setEnabled(False)
        custom_layout.addWidget(self.custom_model_input)
        
        self.browse_model_btn = QPushButton("Browse...")
        self.browse_model_btn.setStyleSheet(self.theme_manager.get_theme().get_button_stylesheet())
        self.browse_model_btn.clicked.connect(self.browse_model)
        self.browse_model_btn.setEnabled(False)
        custom_layout.addWidget(self.browse_model_btn)
        model_layout.addLayout(custom_layout)
        
        model_group.setLayout(model_layout)
        layout.addWidget(model_group)
        
        # Model Info Group
        info_group = QGroupBox("Model Information")
        info_group.setStyleSheet(self.theme_manager.get_theme().get_groupbox_stylesheet())
        info_layout = QVBoxLayout()
        
        self.model_info_text = QTextEdit()
        self.model_info_text.setReadOnly(True)
        self.model_info_text.setMaximumHeight(200)
        self.update_model_info()
        info_layout.addWidget(self.model_info_text)
        
        info_group.setLayout(info_layout)
        layout.addWidget(info_group)
        
        # Performance Comparison Table
        perf_group = QGroupBox("Model Performance Comparison (on COCO)")
        perf_group.setStyleSheet(self.theme_manager.get_theme().get_groupbox_stylesheet())
        perf_layout = QVBoxLayout()
        
        self.perf_table = QTableWidget()
        self.perf_table.setStyleSheet(self.theme_manager.get_theme().get_table_stylesheet())
        self.perf_table.setColumnCount(5)
        self.perf_table.setHorizontalHeaderLabels([
            "Model", "mAP50-95", "Speed (ms)", "Params (M)", "Size (MB)"
        ])
        self.perf_table.horizontalHeader().setStretchLastSection(True)
        
        # Sample data
        models_data = [
            ("YOLOv11n", "39.5", "1.5", "2.6", "5.0"),
            ("YOLOv11s", "47.0", "2.5", "9.4", "18.0"),
            ("YOLOv11m", "51.5", "4.0", "20.1", "38.0"),
            ("YOLOv11l", "53.4", "6.0", "25.3", "48.0"),
            ("YOLOv11x", "54.7", "10.0", "56.9", "108.0"),
        ]
        
        self.perf_table.setRowCount(len(models_data))
        for row, data in enumerate(models_data):
            for col, value in enumerate(data):
                self.perf_table.setItem(row, col, QTableWidgetItem(value))
                
        perf_layout.addWidget(self.perf_table)
        perf_group.setLayout(perf_layout)
        layout.addWidget(perf_group)
        
        layout.addStretch()
        self.setLayout(layout)
        
    def on_source_changed(self, source):
        """Handle model source change"""
        is_custom = source == "Custom .PT File"
        self.pretrained_combo.setEnabled(not is_custom)
        self.custom_model_input.setEnabled(is_custom)
        self.browse_model_btn.setEnabled(is_custom)
        
        if not is_custom:
            # Update pretrained options based on source
            self.pretrained_combo.clear()
            if "YOLOv11" in source:
                models = [
                    "yolo11n.pt (Nano - Fastest)",
                    "yolo11s.pt (Small - Balanced)",
                    "yolo11m.pt (Medium - Accurate)",
                    "yolo11l.pt (Large - More Accurate)",
                    "yolo11x.pt (XLarge - Most Accurate)"
                ]
            elif "YOLOv8" in source:
                models = [
                    "yolov8n.pt (Nano)",
                    "yolov8s.pt (Small)",
                    "yolov8m.pt (Medium)",
                    "yolov8l.pt (Large)",
                    "yolov8x.pt (XLarge)"
                ]
            else:  # YOLOv10
                models = [
                    "yolov10n.pt (Nano)",
                    "yolov10s.pt (Small)",
                    "yolov10m.pt (Medium)",
                    "yolov10l.pt (Large)",
                    "yolov10x.pt (XLarge)"
                ]
            self.pretrained_combo.addItems(models)
            
        self.update_model_info()
        
    def browse_model(self):
        """Browse for custom .pt model file"""
        file, _ = QFileDialog.getOpenFileName(
            self, "Select Custom Model", "", "PyTorch Files (*.pt *.pth)"
        )
        if file:
            self.custom_model_input.setText(file)
            self.selected_model = file
            self.update_model_info()
            
    def get_selected_model(self):
        """Get the selected model path"""
        if self.model_source_combo.currentText() == "Custom .PT File":
            return self.custom_model_input.text()
        else:
            model_text = self.pretrained_combo.currentText()
            return model_text.split()[0]  # Extract just the model name
            
    def update_model_info(self):
        """Update model information display"""
        model = self.get_selected_model()
        
        info_text = f"<b>Selected Model:</b> {model}<br><br>"
        
        if "yolo11n" in model.lower() or "nano" in self.pretrained_combo.currentText().lower():
            info_text += """
            <b>Recommended Use:</b> Jetson Orin Nano, Edge Devices<br>
            <b>Speed:</b> Very Fast (~80 FPS on Jetson Orin Nano)<br>
            <b>Accuracy:</b> Good for most applications<br>
            <b>Best For:</b> Real-time detection, resource-constrained devices
            """
        elif "yolo11s" in model.lower() or "small" in self.pretrained_combo.currentText().lower():
            info_text += """
            <b>Recommended Use:</b> Balanced performance and accuracy<br>
            <b>Speed:</b> Fast (~50 FPS on Jetson Orin Nano)<br>
            <b>Accuracy:</b> Better than nano<br>
            <b>Best For:</b> Production environments with accuracy requirements
            """
        elif "yolo11m" in model.lower() or "medium" in self.pretrained_combo.currentText().lower():
            info_text += """
            <b>Recommended Use:</b> High accuracy applications<br>
            <b>Speed:</b> Medium (~30 FPS on Jetson Orin Nano)<br>
            <b>Accuracy:</b> High accuracy<br>
            <b>Best For:</b> Quality inspection, detailed detection
            """
        else:
            info_text += """
            <b>Custom or Large Model</b><br>
            Please ensure your hardware can support this model.<br>
            Larger models require more memory and processing power.
            """
            
        self.model_info_text.setHtml(info_text)


class TrainingConfigTab(QWidget):
    """Tab 3: Training Configuration"""
    
    def __init__(self, theme_manager):
        super().__init__()
        self.theme_manager = theme_manager
        self.init_ui()
        
    def init_ui(self):
        layout = QVBoxLayout()
        
        # Basic Parameters Group
        basic_group = QGroupBox("Basic Training Parameters")
        basic_group.setStyleSheet(self.theme_manager.get_theme().get_groupbox_stylesheet())
        basic_layout = QGridLayout()
        
        row = 0
        basic_layout.addWidget(QLabel("Epochs:"), row, 0)
        self.epochs_spin = QSpinBox()
        self.epochs_spin.setRange(1, 1000)
        self.epochs_spin.setValue(100)
        self.epochs_spin.setToolTip("Number of training iterations")
        basic_layout.addWidget(self.epochs_spin, row, 1)
        
        basic_layout.addWidget(QLabel("Image Size:"), row, 2)
        self.imgsz_combo = QComboBox()
        self.imgsz_combo.addItems(["320", "416", "512", "640", "800", "1024", "1280"])
        self.imgsz_combo.setCurrentText("640")
        self.imgsz_combo.setToolTip("Input image size (higher = more accurate but slower)")
        basic_layout.addWidget(self.imgsz_combo, row, 3)
        
        row += 1
        basic_layout.addWidget(QLabel("Batch Size:"), row, 0)
        self.batch_spin = QSpinBox()
        self.batch_spin.setRange(-1, 128)
        self.batch_spin.setValue(16)
        self.batch_spin.setSpecialValueText("Auto")
        self.batch_spin.setToolTip("Batch size (-1 for auto, reduce if out of memory)")
        basic_layout.addWidget(self.batch_spin, row, 1)
        
        basic_layout.addWidget(QLabel("Device:"), row, 2)
        self.device_combo = QComboBox()
        self.device_combo.addItems(["0 (GPU 0)", "1 (GPU 1)", "cpu"])
        self.device_combo.setCurrentText("cpu")
        self.device_combo.setToolTip("Computing device (GPU recommended)")
        basic_layout.addWidget(self.device_combo, row, 3)
        
        row += 1
        basic_layout.addWidget(QLabel("Patience:"), row, 0)
        self.patience_spin = QSpinBox()
        self.patience_spin.setRange(0, 200)
        self.patience_spin.setValue(50)
        self.patience_spin.setToolTip("Early stopping patience (epochs without improvement)")
        basic_layout.addWidget(self.patience_spin, row, 1)
        
        basic_layout.addWidget(QLabel("Workers:"), row, 2)
        self.workers_spin = QSpinBox()
        self.workers_spin.setRange(0, 32)
        self.workers_spin.setValue(8)
        self.workers_spin.setToolTip("Number of data loading threads")
        basic_layout.addWidget(self.workers_spin, row, 3)
        
        basic_group.setLayout(basic_layout)
        layout.addWidget(basic_group)
        
        # Optimization Parameters Group
        opt_group = QGroupBox("Optimization Parameters")
        opt_group.setStyleSheet(self.theme_manager.get_theme().get_groupbox_stylesheet())
        opt_layout = QGridLayout()
        
        row = 0
        opt_layout.addWidget(QLabel("Optimizer:"), row, 0)
        self.optimizer_combo = QComboBox()
        self.optimizer_combo.addItems(["auto", "SGD", "Adam", "AdamW", "NAdam", "RAdam", "RMSProp"])
        self.optimizer_combo.setToolTip("Optimization algorithm")
        opt_layout.addWidget(self.optimizer_combo, row, 1)
        
        opt_layout.addWidget(QLabel("Learning Rate:"), row, 2)
        self.lr_spin = QDoubleSpinBox()
        self.lr_spin.setRange(0.0001, 0.1)
        self.lr_spin.setValue(0.01)
        self.lr_spin.setDecimals(4)
        self.lr_spin.setSingleStep(0.001)
        self.lr_spin.setToolTip("Initial learning rate")
        opt_layout.addWidget(self.lr_spin, row, 3)
        
        row += 1
        opt_layout.addWidget(QLabel("Weight Decay:"), row, 0)
        self.weight_decay_spin = QDoubleSpinBox()
        self.weight_decay_spin.setRange(0.0, 0.01)
        self.weight_decay_spin.setValue(0.0005)
        self.weight_decay_spin.setDecimals(5)
        self.weight_decay_spin.setSingleStep(0.0001)
        opt_layout.addWidget(self.weight_decay_spin, row, 1)
        
        opt_layout.addWidget(QLabel("Warmup Epochs:"), row, 2)
        self.warmup_spin = QDoubleSpinBox()
        self.warmup_spin.setRange(0, 10)
        self.warmup_spin.setValue(3.0)
        self.warmup_spin.setToolTip("Learning rate warmup epochs")
        opt_layout.addWidget(self.warmup_spin, row, 3)
        
        opt_group.setLayout(opt_layout)
        layout.addWidget(opt_group)
        
        # Augmentation Parameters Group
        aug_group = QGroupBox("Data Augmentation")
        aug_group.setStyleSheet(self.theme_manager.get_theme().get_groupbox_stylesheet())
        aug_layout = QGridLayout()
        
        row = 0
        aug_layout.addWidget(QLabel("Flip Left-Right:"), row, 0)
        self.fliplr_spin = QDoubleSpinBox()
        self.fliplr_spin.setRange(0.0, 1.0)
        self.fliplr_spin.setValue(0.5)
        self.fliplr_spin.setDecimals(2)
        self.fliplr_spin.setSingleStep(0.1)
        aug_layout.addWidget(self.fliplr_spin, row, 1)
        
        aug_layout.addWidget(QLabel("Mosaic:"), row, 2)
        self.mosaic_spin = QDoubleSpinBox()
        self.mosaic_spin.setRange(0.0, 1.0)
        self.mosaic_spin.setValue(1.0)
        self.mosaic_spin.setDecimals(2)
        self.mosaic_spin.setSingleStep(0.1)
        aug_layout.addWidget(self.mosaic_spin, row, 3)
        
        row += 1
        aug_layout.addWidget(QLabel("HSV Hue:"), row, 0)
        self.hsv_h_spin = QDoubleSpinBox()
        self.hsv_h_spin.setRange(0.0, 0.1)
        self.hsv_h_spin.setValue(0.015)
        self.hsv_h_spin.setDecimals(3)
        aug_layout.addWidget(self.hsv_h_spin, row, 1)
        
        aug_layout.addWidget(QLabel("Scale:"), row, 2)
        self.scale_spin = QDoubleSpinBox()
        self.scale_spin.setRange(0.0, 1.0)
        self.scale_spin.setValue(0.5)
        self.scale_spin.setDecimals(2)
        aug_layout.addWidget(self.scale_spin, row, 3)
        
        aug_group.setLayout(aug_layout)
        layout.addWidget(aug_group)
        
        # Output Settings Group
        output_group = QGroupBox("Output Settings")
        output_group.setStyleSheet(self.theme_manager.get_theme().get_groupbox_stylesheet())
        output_layout = QVBoxLayout()
        
        proj_layout = QHBoxLayout()
        proj_layout.addWidget(QLabel("Project Folder:"))
        self.project_input = QLineEdit()
        # Use absolute path to current working directory
        default_project = str(Path.cwd() / "runs" / "train")
        self.project_input.setText(default_project)
        proj_layout.addWidget(self.project_input)
        
        proj_btn = QPushButton("Browse...")
        proj_btn.setStyleSheet(self.theme_manager.get_theme().get_button_stylesheet())
        proj_btn.clicked.connect(self.browse_project)
        proj_layout.addWidget(proj_btn)
        output_layout.addLayout(proj_layout)
        
        name_layout = QHBoxLayout()
        name_layout.addWidget(QLabel("Experiment Name:"))
        self.name_input = QLineEdit()
        self.name_input.setText(f"exp_{datetime.now().strftime('%Y%m%d_%H%M%S')}")
        name_layout.addWidget(self.name_input)
        output_layout.addLayout(name_layout)
        
        # Checkboxes
        self.amp_check = QCheckBox("Use Mixed Precision (AMP)")
        self.amp_check.setChecked(True)
        self.amp_check.setToolTip("Automatic Mixed Precision for faster training")
        output_layout.addWidget(self.amp_check)
        
        self.plots_check = QCheckBox("Generate Plots")
        self.plots_check.setChecked(True)
        output_layout.addWidget(self.plots_check)
        
        output_group.setLayout(output_layout)
        layout.addWidget(output_group)
        
        layout.addStretch()
        self.setLayout(layout)
        
    def browse_project(self):
        """Browse for project output folder"""
        folder = QFileDialog.getExistingDirectory(self, "Select Project Folder")
        if folder:
            self.project_input.setText(folder)
            
    def get_config(self):
        """Get training configuration as dictionary"""
        device_text = self.device_combo.currentText()
        if "GPU 0" in device_text:
            device = 0
        elif "GPU 1" in device_text:
            device = 1
        else:
            device = 'cpu'
        
        # Convert project path to absolute path
        project_path = self.project_input.text()
        if not os.path.isabs(project_path):
            project_path = str(Path.cwd() / project_path)
            
        return {
            'epochs': self.epochs_spin.value(),
            'imgsz': int(self.imgsz_combo.currentText()),
            'batch': self.batch_spin.value(),
            'device': device,
            'patience': self.patience_spin.value(),
            'workers': self.workers_spin.value(),
            'optimizer': self.optimizer_combo.currentText(),
            'lr0': self.lr_spin.value(),
            'weight_decay': self.weight_decay_spin.value(),
            'warmup_epochs': self.warmup_spin.value(),
            'fliplr': self.fliplr_spin.value(),
            'mosaic': self.mosaic_spin.value(),
            'hsv_h': self.hsv_h_spin.value(),
            'scale': self.scale_spin.value(),
            'project': project_path,
            'name': self.name_input.text(),
            'amp': self.amp_check.isChecked(),
            'plots': self.plots_check.isChecked(),
        }


class TrainingTab(QWidget):
    """Tab 4: Training Execution and Monitoring"""
    
    def __init__(self, theme_manager, parent=None):
        super().__init__(parent)
        self.theme_manager = theme_manager
        self.worker = None
        self.training_metrics = []
        self.init_ui()
        
    def init_ui(self):
        layout = QVBoxLayout()
        
        # Control Buttons
        control_layout = QHBoxLayout()
        self.start_btn = QPushButton("Start Training")
        self.start_btn.setStyleSheet(self.theme_manager.get_theme().get_button_stylesheet('success'))
        self.start_btn.clicked.connect(self.start_training)
        control_layout.addWidget(self.start_btn)
        
        self.stop_btn = QPushButton("Stop Training")
        self.stop_btn.setStyleSheet(self.theme_manager.get_theme().get_button_stylesheet('danger'))
        self.stop_btn.setEnabled(False)
        self.stop_btn.clicked.connect(self.stop_training)
        control_layout.addWidget(self.stop_btn)
        
        layout.addLayout(control_layout)
        
        # Progress Group
        progress_group = QGroupBox("Training Progress")
        progress_group.setStyleSheet(self.theme_manager.get_theme().get_groupbox_stylesheet())
        progress_layout = QVBoxLayout()
        
        self.progress_bar = QProgressBar()
        self.progress_bar.setStyleSheet(self.theme_manager.get_theme().get_progressbar_stylesheet())
        self.progress_bar.setTextVisible(True)
        progress_layout.addWidget(self.progress_bar)
        
        self.status_label = QLabel("Status: Ready")
        self.status_label.setStyleSheet(self.theme_manager.get_theme().get_status_label_stylesheet('info'))
        progress_layout.addWidget(self.status_label)
        
        progress_group.setLayout(progress_layout)
        layout.addWidget(progress_group)
        
        # Metrics Display
        metrics_group = QGroupBox("Current Epoch Metrics")
        metrics_group.setStyleSheet(self.theme_manager.get_theme().get_groupbox_stylesheet())
        metrics_layout = QGridLayout()
        
        self.epoch_label = QLabel("Epoch: 0/0")
        metrics_layout.addWidget(self.epoch_label, 0, 0)
        
        self.box_loss_label = QLabel("Box Loss: -")
        metrics_layout.addWidget(self.box_loss_label, 0, 1)
        
        self.cls_loss_label = QLabel("Cls Loss: -")
        metrics_layout.addWidget(self.cls_loss_label, 0, 2)
        
        self.precision_label = QLabel("Precision: -")
        metrics_layout.addWidget(self.precision_label, 1, 0)
        
        self.recall_label = QLabel("Recall: -")
        metrics_layout.addWidget(self.recall_label, 1, 1)
        
        self.map_label = QLabel("mAP50: -")
        metrics_layout.addWidget(self.map_label, 1, 2)
        
        metrics_group.setLayout(metrics_layout)
        layout.addWidget(metrics_group)
        
        # Live Charts
        if MATPLOTLIB_AVAILABLE:
            chart_group = QGroupBox("Training Metrics (Live)")
            chart_group.setStyleSheet(self.theme_manager.get_theme().get_groupbox_stylesheet())
            chart_layout = QVBoxLayout()
            
            self.figure = Figure(figsize=(10, 4))
            self.canvas = FigureCanvas(self.figure)
            chart_layout.addWidget(self.canvas)
            
            chart_group.setLayout(chart_layout)
            layout.addWidget(chart_group)
            
            # Initialize empty chart
            self.init_chart()
        
        # Training Log
        log_group = QGroupBox("Training Log")
        log_group.setStyleSheet(self.theme_manager.get_theme().get_groupbox_stylesheet())
        log_layout = QVBoxLayout()
        
        self.log_text = QTextEdit()
        self.log_text.setReadOnly(True)
        self.log_text.setMaximumHeight(150)
        log_layout.addWidget(self.log_text)
        
        log_group.setLayout(log_layout)
        layout.addWidget(log_group)
        
        self.setLayout(layout)
        
    def init_chart(self):
        """Initialize empty training chart"""
        if not MATPLOTLIB_AVAILABLE:
            return
            
        self.figure.clear()
        
        self.ax1 = self.figure.add_subplot(131)
        self.ax2 = self.figure.add_subplot(132)
        self.ax3 = self.figure.add_subplot(133)
        
        self.ax1.set_title('Loss')
        self.ax1.set_xlabel('Epoch')
        self.ax1.set_ylabel('Loss')
        self.ax1.grid(True, alpha=0.3)
        
        self.ax2.set_title('Precision & Recall')
        self.ax2.set_xlabel('Epoch')
        self.ax2.set_ylabel('Value')
        self.ax2.grid(True, alpha=0.3)
        
        self.ax3.set_title('mAP')
        self.ax3.set_xlabel('Epoch')
        self.ax3.set_ylabel('mAP')
        self.ax3.grid(True, alpha=0.3)
        
        self.figure.tight_layout()
        self.canvas.draw()
        
    def start_training(self):
        """Start training in worker thread"""
        # Get parent window to access other tabs
        main_window = self.window()
        if not hasattr(main_window, 'dataset_tab'):
            QMessageBox.warning(self, "Error", "Cannot access dataset configuration!")
            return
            
        # Validate inputs
        dataset_tab = main_window.dataset_tab
        model_tab = main_window.model_tab
        config_tab = main_window.config_tab
        
        if not dataset_tab.data_yaml_path:
            QMessageBox.warning(self, "Warning", "Please select dataset in Dataset tab first!")
            main_window.tabs.setCurrentIndex(0)
            return
            
        selected_model = model_tab.get_selected_model()
        if not selected_model:
            QMessageBox.warning(self, "Warning", "Please select a model in Model Selection tab!")
            main_window.tabs.setCurrentIndex(1)
            return
            
        # Prepare training configuration
        training_config = config_tab.get_config()
        training_config['data_yaml'] = dataset_tab.data_yaml_path
        training_config['pretrained_model'] = selected_model
        
        # Reset UI
        self.training_metrics = []
        self.progress_bar.setValue(0)
        self.log_text.clear()
        self.init_chart()
        
        # Create and start worker
        self.worker = TrainingWorker(training_config)
        self.worker.progress_update.connect(self.on_progress_update)
        self.worker.epoch_update.connect(self.on_epoch_update)
        self.worker.status_update.connect(self.on_status_update)
        self.worker.training_complete.connect(self.on_training_complete)
        self.worker.training_error.connect(self.on_training_error)
        
        self.worker.start()
        
        # Update UI state
        self.start_btn.setEnabled(False)
        self.stop_btn.setEnabled(True)
        self.status_label.setText("Status: Training...")
        self.status_label.setStyleSheet(self.theme_manager.get_theme().get_status_label_stylesheet('warning'))
        
        self.log("Training started!")
        
    def stop_training(self):
        """Stop training"""
        if self.worker:
            self.worker.stop()
            self.log("Stopping training...")
            
    @Slot(int)
    def on_progress_update(self, progress):
        """Update progress bar"""
        self.progress_bar.setValue(progress)
        
    @Slot(dict)
    def on_epoch_update(self, metrics):
        """Update metrics display and chart"""
        self.training_metrics.append(metrics)
        
        # Update labels
        self.epoch_label.setText(f"Epoch: {metrics['epoch']}/{metrics['epochs']}")
        self.box_loss_label.setText(f"Box Loss: {metrics['box_loss']:.4f}")
        self.cls_loss_label.setText(f"Cls Loss: {metrics['cls_loss']:.4f}")
        self.precision_label.setText(f"Precision: {metrics['precision']:.4f}")
        self.recall_label.setText(f"Recall: {metrics['recall']:.4f}")
        self.map_label.setText(f"mAP50: {metrics['mAP50']:.4f}")
        
        # Update chart
        self.update_chart()
        
        # Log epoch
        self.log(f"Epoch {metrics['epoch']}/{metrics['epochs']} - "
                f"Loss: {metrics['box_loss']:.4f}, mAP50: {metrics['mAP50']:.4f}")
        
    @Slot(str)
    def on_status_update(self, status):
        """Update status message"""
        self.log(status)
        
    @Slot(str)
    def on_training_complete(self, model_path):
        """Handle training completion"""
        self.start_btn.setEnabled(True)
        self.stop_btn.setEnabled(False)
        self.status_label.setText("Status: Completed!")
        self.status_label.setStyleSheet(self.theme_manager.get_theme().get_status_label_stylesheet('success'))
        self.progress_bar.setValue(100)
        
        self.log(f"Training completed successfully!")
        self.log(f"Best model saved at: {model_path}")
        
        # Update results tab if available
        main_window = self.window()
        if hasattr(main_window, 'results_tab'):
            main_window.results_tab.set_model_path(model_path)
            
        QMessageBox.information(
            self, "Training Complete",
            f"Training completed successfully!\n\nBest model saved at:\n{model_path}"
        )
        
    @Slot(str)
    def on_training_error(self, error_msg):
        """Handle training error"""
        self.start_btn.setEnabled(True)
        self.stop_btn.setEnabled(False)
        self.status_label.setText("Status: Error!")
        self.status_label.setStyleSheet(self.theme_manager.get_theme().get_status_label_stylesheet('error'))
        
        self.log(f"ERROR: {error_msg}")
        QMessageBox.critical(self, "Training Error", f"Training failed:\n{error_msg}")
        
    def update_chart(self):
        """Update training metrics chart"""
        if not MATPLOTLIB_AVAILABLE or not self.training_metrics:
            return
            
        epochs = [m['epoch'] for m in self.training_metrics]
        box_loss = [m['box_loss'] for m in self.training_metrics]
        cls_loss = [m['cls_loss'] for m in self.training_metrics]
        precision = [m['precision'] for m in self.training_metrics]
        recall = [m['recall'] for m in self.training_metrics]
        map50 = [m['mAP50'] for m in self.training_metrics]
        map50_95 = [m['mAP50-95'] for m in self.training_metrics]
        
        # Clear and replot
        self.ax1.clear()
        self.ax2.clear()
        self.ax3.clear()
        
        # Loss plot
        self.ax1.plot(epochs, box_loss, 'b-', label='Box Loss', linewidth=2)
        self.ax1.plot(epochs, cls_loss, 'r-', label='Cls Loss', linewidth=2)
        self.ax1.set_title('Loss')
        self.ax1.set_xlabel('Epoch')
        self.ax1.set_ylabel('Loss')
        self.ax1.legend()
        self.ax1.grid(True, alpha=0.3)
        
        # Precision & Recall plot
        self.ax2.plot(epochs, precision, 'g-', label='Precision', linewidth=2)
        self.ax2.plot(epochs, recall, 'm-', label='Recall', linewidth=2)
        self.ax2.set_title('Precision & Recall')
        self.ax2.set_xlabel('Epoch')
        self.ax2.set_ylabel('Value')
        self.ax2.legend()
        self.ax2.grid(True, alpha=0.3)
        
        # mAP plot
        self.ax3.plot(epochs, map50, 'c-', label='mAP50', linewidth=2)
        self.ax3.plot(epochs, map50_95, 'y-', label='mAP50-95', linewidth=2)
        self.ax3.set_title('mAP')
        self.ax3.set_xlabel('Epoch')
        self.ax3.set_ylabel('mAP')
        self.ax3.legend()
        self.ax3.grid(True, alpha=0.3)
        
        self.figure.tight_layout()
        self.canvas.draw()
        
    def log(self, message):
        """Add message to training log"""
        timestamp = datetime.now().strftime("%H:%M:%S")
        self.log_text.append(f"[{timestamp}] {message}")
        self.log_text.moveCursor(QTextCursor.End)


class ResultsTab(QWidget):
    """Tab 5: Results and Export"""
    
    def __init__(self, theme_manager):
        super().__init__()
        self.theme_manager = theme_manager
        self.model_path = None
        self.init_ui()
        
    def init_ui(self):
        layout = QVBoxLayout()
        
        # Model Info Group
        model_group = QGroupBox("Trained Model")
        model_group.setStyleSheet(self.theme_manager.get_theme().get_groupbox_stylesheet())
        model_layout = QVBoxLayout()
        
        path_layout = QHBoxLayout()
        path_layout.addWidget(QLabel("Model Path:"))
        self.model_path_input = QLineEdit()
        self.model_path_input.setReadOnly(True)
        self.model_path_input.setPlaceholderText("No trained model yet...")
        path_layout.addWidget(self.model_path_input)
        
        browse_btn = QPushButton("Browse...")
        browse_btn.setStyleSheet(self.theme_manager.get_theme().get_button_stylesheet())
        browse_btn.clicked.connect(self.browse_model)
        path_layout.addWidget(browse_btn)
        model_layout.addLayout(path_layout)
        
        # Action buttons
        btn_layout = QHBoxLayout()
        
        self.test_btn = QPushButton("Test on Image")
        self.test_btn.setStyleSheet(self.theme_manager.get_theme().get_button_stylesheet('primary'))
        self.test_btn.clicked.connect(self.test_image)
        btn_layout.addWidget(self.test_btn)
        
        self.validate_btn = QPushButton("Validate Model")
        self.validate_btn.setStyleSheet(self.theme_manager.get_theme().get_button_stylesheet('primary'))
        self.validate_btn.clicked.connect(self.validate_model)
        btn_layout.addWidget(self.validate_btn)
        
        model_layout.addLayout(btn_layout)
        model_group.setLayout(model_layout)
        layout.addWidget(model_group)
        
        # Export Group
        export_group = QGroupBox("Export Model")
        export_group.setStyleSheet(self.theme_manager.get_theme().get_groupbox_stylesheet())
        export_layout = QVBoxLayout()
        
        export_layout.addWidget(QLabel("Export trained model to different formats:"))
        
        export_btn_layout = QHBoxLayout()
        
        self.export_onnx_btn = QPushButton("Export to ONNX")
        self.export_onnx_btn.setStyleSheet(self.theme_manager.get_theme().get_button_stylesheet())
        self.export_onnx_btn.clicked.connect(lambda: self.export_model('onnx'))
        export_btn_layout.addWidget(self.export_onnx_btn)
        
        self.export_engine_btn = QPushButton("Export to TensorRT")
        self.export_engine_btn.setStyleSheet(self.theme_manager.get_theme().get_button_stylesheet())
        self.export_engine_btn.clicked.connect(lambda: self.export_model('engine'))
        export_btn_layout.addWidget(self.export_engine_btn)
        
        self.export_tflite_btn = QPushButton("Export to TFLite")
        self.export_tflite_btn.setStyleSheet(self.theme_manager.get_theme().get_button_stylesheet())
        self.export_tflite_btn.clicked.connect(lambda: self.export_model('tflite'))
        export_btn_layout.addWidget(self.export_tflite_btn)
        
        export_layout.addLayout(export_btn_layout)
        
        # Export options
        opt_layout = QGridLayout()
        
        opt_layout.addWidget(QLabel("Image Size:"), 0, 0)
        self.export_imgsz_combo = QComboBox()
        self.export_imgsz_combo.addItems(["640", "416", "512", "800", "1024"])
        opt_layout.addWidget(self.export_imgsz_combo, 0, 1)
        
        opt_layout.addWidget(QLabel("Precision:"), 0, 2)
        self.export_half_check = QCheckBox("FP16 (Half Precision)")
        self.export_half_check.setChecked(True)
        opt_layout.addWidget(self.export_half_check, 0, 3)
        
        export_layout.addLayout(opt_layout)
        
        export_group.setLayout(export_layout)
        layout.addWidget(export_group)
        
        # Prediction Image Display
        image_group = QGroupBox("Prediction Result")
        image_group.setStyleSheet(self.theme_manager.get_theme().get_groupbox_stylesheet())
        image_layout = QVBoxLayout()
        
        # Scroll area for image
        scroll_area = QScrollArea()
        scroll_area.setWidgetResizable(True)
        scroll_area.setMinimumHeight(300)
        
        self.result_image_label = QLabel()
        self.result_image_label.setAlignment(Qt.AlignCenter)
        self.result_image_label.setText("No prediction result yet. Run 'Test on Image' to see results here.")
        self.result_image_label.setStyleSheet("QLabel { background-color: #F0F0F0; border: 2px dashed #CCCCCC; padding: 20px; }")
        
        scroll_area.setWidget(self.result_image_label)
        image_layout.addWidget(scroll_area)
        
        image_group.setLayout(image_layout)
        layout.addWidget(image_group)
        
        # Validation Results
        results_group = QGroupBox("Validation Results")
        results_group.setStyleSheet(self.theme_manager.get_theme().get_groupbox_stylesheet())
        results_layout = QVBoxLayout()
        
        self.results_text = QTextEdit()
        self.results_text.setReadOnly(True)
        self.results_text.setPlaceholderText("Run validation to see results...")
        results_layout.addWidget(self.results_text)
        
        results_group.setLayout(results_layout)
        layout.addWidget(results_group)
        
        layout.addStretch()
        self.setLayout(layout)
        
    def set_model_path(self, path):
        """Set the trained model path"""
        self.model_path = path
        self.model_path_input.setText(path)
        
    def browse_model(self):
        """Browse for model file"""
        file, _ = QFileDialog.getOpenFileName(
            self, "Select Trained Model", "", "PyTorch Files (*.pt *.pth)"
        )
        if file:
            self.set_model_path(file)
            
    def test_image(self):
        """Test model on a single image"""
        if not self.model_path:
            QMessageBox.warning(self, "Warning", "Please select a trained model first!")
            return
            
        image_file, _ = QFileDialog.getOpenFileName(
            self, "Select Test Image", "",
            "Images (*.jpg *.jpeg *.png *.bmp *.webp)"
        )
        
        if not image_file:
            return
            
        try:
            from ultralytics import YOLO  # type: ignore
            from pathlib import Path
            import os
            
            # Get project folder and experiment name from config tab
            main_window = self.window()
            if hasattr(main_window, 'config_tab'):
                config = main_window.config_tab.get_config()
                project_folder = config.get('project', str(Path.cwd() / "runs" / "train"))
                experiment_name = config.get('name', 'exp')
            else:
                project_folder = str(Path.cwd() / "runs" / "train")
                experiment_name = 'exp'
            
            # Handle model loading with auto-download
            if not os.path.isabs(self.model_path) and not Path(self.model_path).exists():
                QMessageBox.information(self, "Downloading Model", 
                    f"Downloading pre-trained model {self.model_path}...\nThis may take a moment.")
                try:
                    model = YOLO(self.model_path)
                except Exception:
                    # Try without .pt extension
                    model = YOLO(self.model_path.replace('.pt', ''))
            else:
                model = YOLO(self.model_path)
            
            # Run prediction with project folder and experiment name
            results = model.predict(image_file, save=True, conf=0.5, 
                                   project=str(Path(project_folder) / experiment_name),
                                   name='detect_predict')
            
            # Get save path
            save_dir = results[0].save_dir if hasattr(results[0], 'save_dir') else str(Path(project_folder) / experiment_name / 'detect_predict')
            
            # Find and display the predicted image
            predicted_image_path = None
            if save_dir:
                # Look for the predicted image (usually has the same name as input)
                image_name = Path(image_file).name
                predicted_image_path = Path(save_dir) / image_name
                
                if predicted_image_path.exists():
                    # Load and display the image
                    pixmap = QPixmap(str(predicted_image_path))
                    if not pixmap.isNull():
                        # Scale image to fit display (max width 800px, maintain aspect ratio)
                        scaled_pixmap = pixmap.scaledToWidth(800, Qt.SmoothTransformation)
                        self.result_image_label.setPixmap(scaled_pixmap)
                        self.result_image_label.setStyleSheet("QLabel { background-color: #FFFFFF; border: 2px solid #CCCCCC; }")
                    else:
                        self.result_image_label.setText(f"Failed to load image from:\n{predicted_image_path}")
                else:
                    self.result_image_label.setText(f"Predicted image not found at:\n{predicted_image_path}")
            
            # Create custom message box with Open Directory button
            msg_box = QMessageBox(self)
            msg_box.setWindowTitle("Success")
            msg_box.setText("Prediction completed!")
            msg_box.setInformativeText(f"Results saved to:\n{save_dir}")
            msg_box.setIcon(QMessageBox.Information)
            
            # Add Open Directory button on the left
            open_btn = msg_box.addButton("Open Directory", QMessageBox.ActionRole)
            
            # Show message box and handle button clicks
            msg_box.exec()
            if msg_box.clickedButton() == open_btn:
                # Open directory in file explorer
                try:
                    if save_dir and isinstance(save_dir, str):
                        if platform.system() == 'Windows':
                            os.startfile(save_dir)
                        elif platform.system() == 'Darwin':
                            subprocess.Popen(['open', save_dir])
                        else:  # Linux
                            subprocess.Popen(['xdg-open', save_dir])
                    else:
                        QMessageBox.warning(self, "Error", "Invalid results directory path.")
                except Exception as e:
                    QMessageBox.warning(self, "Error", f"Failed to open directory:\n{str(e)}")
            
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Failed to run prediction:\n{str(e)}")
            
    def validate_model(self):
        """Validate model on dataset"""
        if not self.model_path:
            QMessageBox.warning(self, "Warning", "Please select a trained model first!")
            return
            
        # Get dataset from dataset tab
        main_window = self.window()
        if not hasattr(main_window, 'dataset_tab'):
            QMessageBox.warning(self, "Error", "Cannot access dataset configuration!")
            return
            
        data_yaml = main_window.dataset_tab.data_yaml_path
        if not data_yaml:
            QMessageBox.warning(self, "Warning", "Please configure dataset first!")
            return
            
        try:
            from ultralytics import YOLO  # type: ignore
            from pathlib import Path
            import os
            
            # Handle model loading with auto-download
            if not os.path.isabs(self.model_path) and not Path(self.model_path).exists():
                self.results_text.setText(f"Downloading pre-trained model {self.model_path}... Please wait.")
                QApplication.processEvents()
                try:
                    model = YOLO(self.model_path)
                except Exception:
                    # Try without .pt extension
                    model = YOLO(self.model_path.replace('.pt', ''))
            else:
                self.results_text.setText("Running validation... Please wait.")
                QApplication.processEvents()
                model = YOLO(self.model_path)
            metrics = model.val(data=data_yaml) if data_yaml else None
            if not metrics:
                self.results_text.setText("Validation failed: data_yaml not provided.")
                return
            
            # Format results
            results_text = f"""
<h3>Validation Results</h3>
<table border='1' cellpadding='5' cellspacing='0'>
<tr><td><b>Metric</b></td><td><b>Value</b></td></tr>
<tr><td>mAP50</td><td>{metrics.box.map50:.4f}</td></tr>
<tr><td>mAP50-95</td><td>{metrics.box.map:.4f}</td></tr>
<tr><td>Precision</td><td>{metrics.box.mp:.4f}</td></tr>
<tr><td>Recall</td><td>{metrics.box.mr:.4f}</td></tr>
<tr><td>F1 Score</td><td>{2 * (metrics.box.mp * metrics.box.mr) / (metrics.box.mp + metrics.box.mr) if (metrics.box.mp + metrics.box.mr) > 0 else 0:.4f}</td></tr>
</table>
<br>
<p><i>Results saved in runs/detect/val</i></p>
            """
            
            self.results_text.setHtml(results_text)
            
        except Exception as e:
            self.results_text.setText(f"Validation failed:\n{str(e)}")
            QMessageBox.critical(self, "Error", f"Failed to validate model:\n{str(e)}")
            
    def export_model(self, format_type):
        """Export model to specified format"""
        if not self.model_path:
            QMessageBox.warning(self, "Warning", "Please select a trained model first!")
            return
            
        try:
            from ultralytics import YOLO  # type: ignore
            from pathlib import Path
            import os
            
            # Handle model loading with auto-download
            if not os.path.isabs(self.model_path) and not Path(self.model_path).exists():
                self.results_text.setText(f"Downloading pre-trained model {self.model_path}... Please wait.")
                QApplication.processEvents()
                try:
                    model = YOLO(self.model_path)
                except Exception:
                    # Try without .pt extension
                    model = YOLO(self.model_path.replace('.pt', ''))
            else:
                model = YOLO(self.model_path)
            
            imgsz = int(self.export_imgsz_combo.currentText())
            half = self.export_half_check.isChecked()
            
            self.results_text.setText(f"Exporting to {format_type.upper()}... Please wait.")
            QApplication.processEvents()
            
            if format_type == 'engine':
                # TensorRT export
                export_path = model.export(format='engine', imgsz=imgsz, half=half, device=0)
            else:
                # ONNX or TFLite export
                export_path = model.export(format=format_type, imgsz=imgsz, half=half)
                
            QMessageBox.information(
                self, "Success",
                f"Model exported successfully!\n\nSaved to:\n{export_path}"
            )
            
            self.results_text.append(f"\n\nExport successful: {export_path}")
            
        except Exception as e:
            self.results_text.append(f"\n\nExport failed: {str(e)}")
            QMessageBox.critical(self, "Error", f"Failed to export model:\n{str(e)}")


class YOLOTrainerMainWindow(QMainWindow):
    """Main application window"""
    
    def __init__(self):
        super().__init__()
        self.setWindowTitle("YOLO Training Tool")
        self.setGeometry(100, 100, 1400, 900)
        
        # Set application icon
        icon_path = Path(__file__).parent / 'logo.png'
        if icon_path.exists():
            self.setWindowIcon(QIcon(str(icon_path)))
        
        # Initialize theme manager with Light theme only
        self.theme_manager = ThemeManager()
        self.theme_manager.set_theme('Light')
        
        # Create central widget and main layout
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        main_layout = QVBoxLayout(central_widget)
        
        # Header (simple, no theme switcher)
        #header = QLabel("YOLO Object Detection - Training Interface")
        #header.setStyleSheet(self.theme_manager.get_theme().get_header_stylesheet())
        #header.setAlignment(Qt.AlignCenter)
        #main_layout.addWidget(header)
        
        # Create tab widget
        self.tabs = QTabWidget()
        self.tabs.setStyleSheet(self.theme_manager.get_theme().get_tab_stylesheet())
        
        # Create tabs with theme manager
        self.dataset_tab = DatasetTab(self.theme_manager)
        self.model_tab = ModelSelectionTab(self.theme_manager)
        self.config_tab = TrainingConfigTab(self.theme_manager)
        self.training_tab = TrainingTab(self.theme_manager, self)
        self.results_tab = ResultsTab(self.theme_manager)
        
        # Add tabs
        self.tabs.addTab(self.dataset_tab, "1. Dataset")
        self.tabs.addTab(self.model_tab, "2. Model Selection")
        self.tabs.addTab(self.config_tab, "3. Configuration")
        self.tabs.addTab(self.training_tab, "4. Training")
        self.tabs.addTab(self.results_tab, "5. Model Result")
        
        main_layout.addWidget(self.tabs)
        
        # Status bar
        self.statusBar().showMessage("Ready - AGC Automotive Thailand Dx")
        
        # Apply Light theme stylesheet
        self.setStyleSheet(self.theme_manager.get_theme().get_main_stylesheet())
        
    def closeEvent(self, event):
        """Handle window close event"""
        if hasattr(self.training_tab, 'worker') and self.training_tab.worker and self.training_tab.worker.isRunning():
            reply = QMessageBox.question(
                self, 'Confirm Exit',
                'Training is in progress. Are you sure you want to exit?',
                QMessageBox.Yes | QMessageBox.No,
                QMessageBox.No
            )
            
            if reply == QMessageBox.Yes:
                self.training_tab.stop_training()
                event.accept()
            else:
                event.ignore()
        else:
            event.accept()


def main():
    """Main application entry point"""
    app = QApplication(sys.argv)
    
    # Set application style
    app.setStyle('Fusion')
    
    # Set application icon for taskbar
    icon_path = Path(__file__).parent / 'logo.png'
    if icon_path.exists():
        app.setWindowIcon(QIcon(str(icon_path)))
    
    # Check ultralytics version
    try:
        import ultralytics
        version = ultralytics.__version__
        print(f"Ultralytics version: {version}")
        
        major, minor = map(int, version.split('.')[:2])
        if major < 8 or (major == 8 and minor < 1):
            msg = QMessageBox()
            msg.setIcon(QMessageBox.Warning)
            msg.setWindowTitle("Ultralytics Version Warning")
            msg.setText(f"Current ultralytics version: {version}")
            msg.setInformativeText(
                "YOLOv11 models require ultralytics>=8.1.0\n\n"
                "You can either:\n"
                "1. Upgrade ultralytics: pip install --upgrade ultralytics\n"
                "2. Use YOLOv8 models instead (yolov8n.pt, yolov8s.pt, etc.)\n\n"
                "The application will continue, but YOLOv11 models may not work."
            )
            msg.setStandardButtons(QMessageBox.Ok)
            msg.exec()
    except Exception as e:
        print(f"Could not check ultralytics version: {e}")
    
    # Create and show main window
    window = YOLOTrainerMainWindow()
    window.show()
    
    sys.exit(app.exec())


if __name__ == '__main__':
    main()
