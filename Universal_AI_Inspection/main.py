import sys
import time
from pathlib import Path
from typing import List, Optional

import cv2
import numpy as np
from PySide6.QtCore import Qt, QThread, Signal, Slot, QSize
from PySide6.QtGui import QBrush, QColor, QFont, QImage, QPainter, QPen, QPixmap, QWheelEvent
from PySide6.QtWidgets import (
    QApplication,
    QCheckBox,
    QComboBox,
    QFileDialog,
    QFrame,
    QHBoxLayout,
    QHeaderView,
    QLabel,
    QLineEdit,
    QPushButton,
    QScrollArea,
    QSlider,
    QSpinBox,
    QTabWidget,
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
    QWidget,
)

try:
    from ultralytics import YOLO  # type: ignore
    YOLO("yolo26n.pt")
except Exception:
    YOLO = None

try:
    from Relay_B import Relay  # type: ignore
    RELAY_AVAILABLE = True
except Exception:
    RELAY_AVAILABLE = False


ALLOWED_FPS = [24, 30, 60]


def _video_backend_for_platform() -> int:
    if sys.platform.startswith("linux"):
        return cv2.CAP_V4L2
    if sys.platform.startswith("win"):
        return cv2.CAP_DSHOW
    if sys.platform.startswith("darwin"):
        # macOS typically uses AVFoundation
        return cv2.CAP_AVFOUNDATION
    return 0


def _cuda_available() -> bool:
    try:
        import torch  # type: ignore
        return torch.cuda.is_available()
    except Exception:
        return False


def find_cameras(max_index: int = 10) -> List[int]:
    available = []
    backend = _video_backend_for_platform()
    for idx in range(max_index):
        try:
            cap = cv2.VideoCapture(idx, backend) if backend != 0 else cv2.VideoCapture(idx)
            if cap is None:
                continue
            if cap.isOpened():
                available.append(idx)
            cap.release()
        except Exception:
            try:
                if cap is not None:
                    cap.release()
            except Exception:
                pass
    return available


def nearest_allowed_fps(value: float) -> int:
    if value <= 1:
        return 30
    return min(ALLOWED_FPS, key=lambda x: abs(x - value))


class ZoomableLabel(QLabel):
    """Custom QLabel that supports mouse wheel zooming."""
    zoom_changed = Signal(float)  # Emits new zoom level
    
    def __init__(self) -> None:
        super().__init__()
        self._zoom_level = 1.0
        self._original_pixmap = None
    
    def set_zoom_level(self, zoom: float) -> None:
        """Set zoom level (1.0 = 100%)."""
        self._zoom_level = max(0.1, min(zoom, 5.0))  # Clamp between 0.1x and 5.0x
        self.zoom_changed.emit(self._zoom_level)
    
    def get_zoom_level(self) -> float:
        return self._zoom_level
    
    def setPixmap(self, pixmap: QPixmap) -> None:
        """Override setPixmap to store original and apply zoom."""
        self._original_pixmap = pixmap
        self._update_display()
    
    def _update_display(self) -> None:
        """Update displayed pixmap with current zoom level."""
        if self._original_pixmap is None or self._original_pixmap.isNull():
            return
        
        # Scale original pixmap by zoom level
        zoomed_size = QSize(
            int(self._original_pixmap.width() * self._zoom_level),
            int(self._original_pixmap.height() * self._zoom_level)
        )
        zoomed_pixmap = self._original_pixmap.scaledToWidth(
            zoomed_size.width(),
            Qt.SmoothTransformation
        )
        super().setPixmap(zoomed_pixmap)
    
    def wheelEvent(self, event: QWheelEvent) -> None:
        """Handle mouse wheel for zooming."""
        if event.angleDelta().y() > 0:
            # Scroll up = zoom in
            self.set_zoom_level(self._zoom_level * 1.1)
        else:
            # Scroll down = zoom out
            self.set_zoom_level(self._zoom_level / 1.1)
        event.accept()



class VideoWorker(QThread):
    frame_ready = Signal(QImage, list)
    status = Signal(str)

    def __init__(self) -> None:
        super().__init__()
        self._running = False
        self._camera_index: Optional[int] = None
        self._model_path: Optional[Path] = None
        self._model = None
        self._target_fps: int = 30
        self._device: str = "cuda" if _cuda_available() else "cpu"

    def set_camera_index(self, index: Optional[int]) -> None:
        self._camera_index = index

    def set_model_path(self, path: Optional[Path]) -> None:
        self._model_path = path
        self._model = None

    def set_device(self, device: str) -> None:
        self._device = device

    def stop(self) -> None:
        self._running = False

    def _load_model(self) -> None:
        if self._model_path is None:
            return
        if YOLO is None:
            self.status.emit("Ultralytics not available. Install requirements.")
            return
        try:
            try:
                self._model = YOLO(str(self._model_path), device=self._device)
            except TypeError:
                # Older ultralytics may not accept device in constructor
                self._model = YOLO(str(self._model_path))
                # Try to move underlying PyTorch model to device if available
                try:
                    import torch  # type: ignore
                    if self._device != "cpu" and hasattr(self._model, "model"):
                        try:
                            self._model.model.to(self._device)
                        except Exception:
                            pass
                except Exception:
                    pass
            self.status.emit(f"Model loaded: {self._model_path.name}")
        except Exception as exc:
            self.status.emit(f"Model load failed: {exc}")
            self._model = None

    def _extract_detections(self, results) -> List[dict]:
        detections: List[dict] = []
        if results is None:
            return detections
        boxes = results.boxes
        names = results.names
        if boxes is None:
            return detections
        for box in boxes:
            xyxy = box.xyxy[0].cpu().numpy().astype(int)
            cls_id = int(box.cls[0].item())
            conf = float(box.conf[0].item()) if box.conf is not None else 0.0
            class_name = str(names.get(cls_id, cls_id))
            label = f"{class_name} {conf:.2f}"
            x1, y1, x2, y2 = xyxy.tolist()
            detections.append(
                {
                    "x1": x1,
                    "y1": y1,
                    "x2": x2,
                    "y2": y2,
                    "label": label,
                    "class_name": class_name,
                }
            )
        return detections

    def run(self) -> None:
        if self._camera_index is None:
            self.status.emit("Select a camera.")
            return

        backend = _video_backend_for_platform()
        cap = cv2.VideoCapture(self._camera_index, backend) if backend != 0 else cv2.VideoCapture(self._camera_index)
        # Fallback: if using a specific backend fails, try default constructor
        if not cap.isOpened() and backend != 0:
            try:
                cap.release()
            except Exception:
                pass
            cap = cv2.VideoCapture(self._camera_index)

        if not cap.isOpened():
            self.status.emit("Camera open failed.")
            return

        fps = cap.get(cv2.CAP_PROP_FPS) or 0
        self._target_fps = nearest_allowed_fps(fps)
        self.status.emit(f"Camera FPS: {fps:.2f} -> Target FPS: {self._target_fps}")

        if self._model_path is not None and self._model is None:
            self._load_model()

        self._running = True
        frame_period = 1.0 / self._target_fps

        while self._running:
            start_time = time.perf_counter()

            ret, frame = cap.read()
            if not ret:
                self.status.emit("Frame grab failed.")
                break

            detections = []
            if self._model is not None:
                try:
                    results = self._model(frame, verbose=False)[0]
                except Exception as exc:
                    self.status.emit(f"Inference error: {exc}")
                    results = None
                detections = self._extract_detections(results)

            elapsed = time.perf_counter() - start_time

            if elapsed < frame_period:
                time.sleep(frame_period - elapsed)
            else:
                frames_to_skip = int(elapsed / frame_period) - 1
                for _ in range(max(0, frames_to_skip)):
                    cap.grab()

            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            h, w, ch = rgb.shape
            bytes_per_line = ch * w
            image = QImage(rgb.data, w, h, bytes_per_line, QImage.Format_RGB888)
            self.frame_ready.emit(image, detections)

        cap.release()
        self.status.emit("Stopped.")


class MainWindow(QWidget):
    def __init__(self) -> None:
        super().__init__()
        self.setWindowTitle("AI Object Detection Inspection")
        
        self._worker = VideoWorker()
        self._worker.frame_ready.connect(self.on_frame)
        self._worker.status.connect(self.on_status)
        self._class_counts = {}
        self._class_colors = {}
        self._current_pixmap = None
        self._selected_classes = set()
        self._confidence_threshold = 0.0
        self._model_classes = []
        self._match_class = None
        self._match_detected = False
        self._zoom_level = 1.0
        
        # Relay control
        self._relay = None
        self._relay_connected = False
        self._relay_host = "192.168.1.201"
        self._relay_port = 502
        # Track 8 class-to-relay-channel mappings
        # last_on_time: timestamp when relay was last turned ON (for minimum hold time)
        self._relay_min_on_seconds = 1.0  # relay stays ON at least this long
        self._relay_mappings = [
            {"class": None, "channel": i + 1, "last_state": False, "last_on_time": 0.0} for i in range(8)
        ]
            
        # Metrics tracking
        self._fps_counter = 0
        self._fps_timer = 0
        self._current_fps = 0
        self._total_detections = 0
        self._inference_time = 0

        # Header Section
        header_frame = QFrame()
        header_layout = QHBoxLayout()
        
        self.model_label = QLabel("No model selected")
        self.model_label.setStyleSheet("font-size: 12pt; font-weight: bold")
        self.load_model_button = QPushButton("📁 Load Model")
        self.load_model_button.clicked.connect(self.load_model)
        self.load_model_button.setMinimumHeight(40)
        
        # Compute device selector
        self.compute_combo = QComboBox()
        self.compute_combo.addItem("cpu")
        self.compute_combo.addItem("cuda")
        self.compute_combo.addItem("cuda:0")
        self.compute_combo.setMaximumWidth(140)
        # Auto-select CUDA if available (connection will be made later after status_label is created)
        try:
            if _cuda_available():
                idx = self.compute_combo.findText("cuda")
                if idx >= 0:
                    self.compute_combo.setCurrentIndex(idx)
                    self._worker.set_device("cuda")
                    # reflect in status label
                    self.status_label.setText("Compute: cuda")
            else:
                self._worker.set_device("cpu")
        except Exception:
            self._worker.set_device("cpu")
        
        header_layout.addWidget(QLabel("Model:"), 0)
        header_layout.addWidget(self.model_label, 1)
        header_layout.addWidget(self.load_model_button, 0)
        header_layout.addWidget(QLabel("Compute:"), 0)
        header_layout.addWidget(self.compute_combo, 0)
        header_frame.setLayout(header_layout)
        
        # Metrics Panel
        metrics_frame = QFrame()
        metrics_layout = QHBoxLayout()
        
        self.fps_label = QLabel("FPS: 0")
        self.fps_label.setStyleSheet("font-size: 11pt; font-weight: bold")
        
        self.detections_label = QLabel("Detections: 0")
        self.detections_label.setStyleSheet("font-size: 11pt; font-weight: bold")
        
        self.status_label = QLabel("Ready")
        self.status_label.setStyleSheet("font-size: 10pt")
        
        # Now safe to connect the signal after status_label exists
        self.compute_combo.currentTextChanged.connect(self.on_compute_changed)
        
        metrics_layout.addWidget(self.fps_label, 0)
        metrics_layout.addWidget(self.detections_label, 0)
        metrics_layout.addWidget(self.status_label, 1)
        metrics_frame.setLayout(metrics_layout)
        
        # Match Status
        self.match_status_label = QLabel("")
        self.match_status_label.setAlignment(Qt.AlignCenter)
        self.match_status_label.setStyleSheet(
            "background-color: #00d9ff; color: #000000; font-size: 14pt; font-weight: bold; "
            "padding: 12px; border-radius: 6px;"
        )
        self.match_status_label.setMinimumHeight(50)
        
        # Camera Controls Section
        cam_frame = QFrame()
        cam_layout = QHBoxLayout()
        
        self.camera_combo = QComboBox()
        self.camera_combo.setMaximumWidth(240)
        self.scan_button = QPushButton("🔄 Scan")
        self.scan_button.clicked.connect(self.scan_cameras)
        self.scan_button.setMaximumWidth(120)
        
        self.start_button = QPushButton("▶ Start")
        self.start_button.clicked.connect(self.start_stream)
        self.start_button.setMaximumWidth(120)
        
        self.stop_button = QPushButton("⏹ Stop")
        self.stop_button.clicked.connect(self.stop_stream)
        self.stop_button.setMaximumWidth(120)
        
        self.capture_button = QPushButton("📷 Capture")
        self.capture_button.clicked.connect(self.capture_frame)
        self.capture_button.setMaximumWidth(120)
        
        cam_layout.addWidget(QLabel("Camera:"), 0)
        cam_layout.addWidget(self.camera_combo, 1)
        cam_layout.addWidget(self.scan_button, 0)
        cam_layout.addWidget(self.start_button, 0)
        cam_layout.addWidget(self.stop_button, 0)
        cam_layout.addWidget(self.capture_button, 0)
        cam_frame.setLayout(cam_layout)
        
        # Video Display with Zoom Controls
        video_frame = QFrame()
        video_layout = QVBoxLayout()
        
        # Zoom controls
        zoom_control_layout = QHBoxLayout()
        self.zoom_out_button = QPushButton("🔍➖")
        self.zoom_out_button.setMaximumWidth(50)
        self.zoom_out_button.clicked.connect(self.zoom_out)
        
        self.zoom_label = QLabel("Zoom: 100%")
        self.zoom_label.setAlignment(Qt.AlignCenter)
        self.zoom_label.setStyleSheet("font-size: 10pt; font-weight: bold;")
        
        self.zoom_in_button = QPushButton("🔍➕")
        self.zoom_in_button.setMaximumWidth(50)
        self.zoom_in_button.clicked.connect(self.zoom_in)
        
        self.reset_zoom_button = QPushButton("Reset")
        self.reset_zoom_button.setMaximumWidth(70)
        self.reset_zoom_button.clicked.connect(self.reset_zoom)
        
        zoom_control_layout.addWidget(self.zoom_out_button, 0)
        zoom_control_layout.addWidget(self.zoom_label, 1)
        zoom_control_layout.addWidget(self.zoom_in_button, 0)
        zoom_control_layout.addWidget(self.reset_zoom_button, 0)
        
        # Video Display
        self.video_label = ZoomableLabel()
        self.video_label.setAlignment(Qt.AlignCenter)
        self.video_label.setMinimumSize(640, 480)
        self.video_label.setStyleSheet("background-color: #000000; border-radius: 8px;")
        self.video_label.zoom_changed.connect(self.on_zoom_changed)
        
        video_layout.addLayout(zoom_control_layout, 0)
        video_layout.addWidget(self.video_label, 1)
        video_frame.setLayout(video_layout)
        
        # Filter Panel (Sidebar)
        filter_frame = QFrame()
        filter_layout = QVBoxLayout()
        
        filter_title = QLabel("🎛️ Filters & Settings")
        filter_title.setStyleSheet("font-size: 12pt; font-weight: bold")
        filter_layout.addWidget(filter_title)
        
        # Confidence Section
        conf_label = QLabel("Confidence Threshold")
        conf_label.setStyleSheet("font-size: 10pt; font-weight: bold;")
        filter_layout.addWidget(conf_label)
        
        self.confidence_slider = QSlider(Qt.Horizontal)
        self.confidence_slider.setMinimum(0)
        self.confidence_slider.setMaximum(100)
        self.confidence_slider.setValue(0)
        self.confidence_slider.setTickPosition(QSlider.TicksBelow)
        self.confidence_slider.setTickInterval(10)
        self.confidence_slider.valueChanged.connect(self.on_confidence_changed)
        filter_layout.addWidget(self.confidence_slider)
        
        self.confidence_label = QLabel("0%")
        #self.confidence_label.setStyleSheet("font-size: 10pt; color: #00ff00; font-weight: bold;")
        self.confidence_label.setAlignment(Qt.AlignCenter)
        filter_layout.addWidget(self.confidence_label)
        
        filter_layout.addSpacing(15)
        
        # Classes Section
        classes_label = QLabel("Classes to Show")
        classes_label.setStyleSheet("font-size: 10pt; font-weight: bold;")
        filter_layout.addWidget(classes_label)
        
        self.class_filters_scroll = QScrollArea()
        self.class_filters_scroll.setWidgetResizable(True)
        self.class_filters_container = QWidget()
        self.class_filters_layout = QVBoxLayout()
        self.class_filters_container.setLayout(self.class_filters_layout)
        self.class_filters_scroll.setWidget(self.class_filters_container)
        self.class_filters_scroll.setMinimumHeight(180)
        filter_layout.addWidget(self.class_filters_scroll, 1)
        
        filter_frame.setLayout(filter_layout)
        filter_frame.setMaximumWidth(240)
        
        # Detection Table
        table_frame = QFrame()
        table_layout = QVBoxLayout()
        
        table_title = QLabel("📊 Detections")
        table_title.setStyleSheet("font-size: 10pt; font-weight: bold; color: #00d9ff;")
        table_layout.addWidget(table_title)
        
        self.table = QTableWidget(0, 2)
        self.table.setHorizontalHeaderLabels(["Class", "Count"])
        self.table.verticalHeader().setVisible(False)
        self.table.setEditTriggers(QTableWidget.NoEditTriggers)
        self.table.setSelectionMode(QTableWidget.NoSelection)
        self.table.setAlternatingRowColors(True)
        table_layout.addWidget(self.table)
        table_frame.setLayout(table_layout)
        table_frame.setMaximumWidth(240)
        
        # ── Build Relay Tab ──
        relay_tab = QWidget()
        relay_tab_layout = QVBoxLayout()
        
        # Relay Connection Section
        relay_conn_frame = QFrame()
        relay_conn_layout = QHBoxLayout()
        
        relay_conn_layout.addWidget(QLabel("Host:"), 0)
        self.relay_host_input = QLineEdit()
        self.relay_host_input.setText(self._relay_host)
        self.relay_host_input.setPlaceholderText("192.168.1.201")
        self.relay_host_input.setMaximumWidth(180)
        relay_conn_layout.addWidget(self.relay_host_input, 0)
        
        relay_conn_layout.addWidget(QLabel("Port:"), 0)
        self.relay_port_spin = QSpinBox()
        self.relay_port_spin.setRange(1, 65535)
        self.relay_port_spin.setValue(self._relay_port)
        self.relay_port_spin.setMaximumWidth(100)
        relay_conn_layout.addWidget(self.relay_port_spin, 0)
        
        self.relay_connect_button = QPushButton("🔌 Connect Relay")
        self.relay_connect_button.clicked.connect(self.connect_relay)
        self.relay_connect_button.setMinimumHeight(36)
        relay_conn_layout.addWidget(self.relay_connect_button, 0)
        
        self.relay_status_label = QLabel("Status: Disconnected")
        self.relay_status_label.setStyleSheet("font-size: 11pt; color: #ff6b6b; font-weight: bold;")
        relay_conn_layout.addWidget(self.relay_status_label, 1)
        
        relay_conn_frame.setLayout(relay_conn_layout)
        relay_tab_layout.addWidget(relay_conn_frame, 0)
        
        # Match-to-Relay Mapping Table (8 rows)
        mapping_label = QLabel("🎯 Class → Relay Channel Mapping")
        mapping_label.setStyleSheet("font-size: 12pt; font-weight: bold;")
        relay_tab_layout.addWidget(mapping_label)
        
        self.relay_mapping_table = QTableWidget(8, 3)
        self.relay_mapping_table.setHorizontalHeaderLabels(["Class", "Relay Channel", "Relay Status"])
        self.relay_mapping_table.verticalHeader().setVisible(False)
        self.relay_mapping_table.setEditTriggers(QTableWidget.NoEditTriggers)
        self.relay_mapping_table.setSelectionMode(QTableWidget.NoSelection)
        self.relay_mapping_table.setAlternatingRowColors(True)
        header = self.relay_mapping_table.horizontalHeader()
        header.setSectionResizeMode(0, QHeaderView.Stretch)
        header.setSectionResizeMode(1, QHeaderView.ResizeToContents)
        header.setSectionResizeMode(2, QHeaderView.ResizeToContents)
        
        # Populate 8 rows with dropdowns
        for row in range(8):
            # Column 0: Class dropdown
            class_combo = QComboBox()
            class_combo.addItem("None", None)
            class_combo.currentIndexChanged.connect(
                lambda idx, r=row: self._on_relay_mapping_class_changed(r)
            )
            self.relay_mapping_table.setCellWidget(row, 0, class_combo)
            
            # Column 1: Relay channel dropdown
            ch_combo = QComboBox()
            for ch in range(1, 9):
                ch_combo.addItem(f"CH {ch}", ch)
            ch_combo.setCurrentIndex(row)  # Default: row 0 = CH1, row 1 = CH2, etc.
            ch_combo.currentIndexChanged.connect(
                lambda idx, r=row: self._on_relay_mapping_channel_changed(r)
            )
            self.relay_mapping_table.setCellWidget(row, 1, ch_combo)
            
            # Column 2: Relay status label
            status_item = QTableWidgetItem("OFF")
            status_item.setTextAlignment(Qt.AlignCenter)
            status_item.setForeground(QBrush(QColor("#ff6b6b")))
            self.relay_mapping_table.setItem(row, 2, status_item)
        
        relay_tab_layout.addWidget(self.relay_mapping_table, 1)
        
        # Match Status (moved here)
        relay_tab_layout.addWidget(self.match_status_label, 0)
        
        relay_tab.setLayout(relay_tab_layout)
        
        # ── Tab Widget to hold Monitor + Relay ──
        self.tab_widget = QTabWidget()
        
        # Monitor tab (main content)
        monitor_tab = QWidget()
        monitor_layout = QHBoxLayout()
        monitor_layout.addWidget(video_frame, 2)
        monitor_layout.addWidget(filter_frame, 0)
        monitor_layout.addWidget(table_frame, 0)
        monitor_tab.setLayout(monitor_layout)
        
        self.tab_widget.addTab(monitor_tab, "📹 Monitor")
        self.tab_widget.addTab(relay_tab, "⚡ Relay")
        
        # Main layout
        layout = QVBoxLayout()
        layout.addWidget(header_frame, 0)
        layout.addWidget(metrics_frame, 0)
        layout.addWidget(cam_frame, 0)
        layout.addWidget(self.tab_widget, 1)
        
        self.setLayout(layout)
        layout.setContentsMargins(12, 12, 12, 12)
        layout.setSpacing(12)

        self.scan_cameras()

    @Slot()
    def scan_cameras(self) -> None:
        self.camera_combo.clear()
        cameras = find_cameras()
        if not cameras:
            self.camera_combo.addItem("No camera", None)
            return
        for idx in cameras:
            self.camera_combo.addItem(f"Camera {idx}", idx)

    @Slot()
    def load_model(self) -> None:
        file_path, _ = QFileDialog.getOpenFileName(
            self,
            "Select YOLO .pt Model",
            "",
            "PyTorch Model (*.pt)",
        )
        if not file_path:
            return
        model_path = Path(file_path)
        self.model_label.setText(model_path.name)
        self._worker.set_model_path(model_path)
        self.status_label.setText("Loading model...")
        
        # Load model to extract class names
        if YOLO is not None:
            try:
                try:
                    model = YOLO(str(model_path), device=self.compute_combo.currentText())
                except TypeError:
                    model = YOLO(str(model_path))
                self._model_classes = list(model.names.values())
                self._populate_class_filters()
                self.status_label.setText(f"Model loaded: {len(self._model_classes)} classes")
            except Exception as exc:
                self.status_label.setText(f"Failed to load model: {exc}")
                self._model_classes = []
        else:
            self.status_label.setText("Ultralytics not available.")
            self._model_classes = []

    def _populate_class_filters(self) -> None:
        """Create checkboxes for all model classes."""
        # Clear existing checkboxes
        while self.class_filters_layout.count() > 0:
            item = self.class_filters_layout.takeAt(0)
            if item.widget():
                item.widget().deleteLater()
        
        self._selected_classes.clear()
        
        # Create checkbox for each class
        for class_name in sorted(self._model_classes):
            checkbox = QCheckBox(class_name)
            checkbox.setChecked(True)
            checkbox.toggled.connect(
                lambda checked, cn=class_name: self._toggle_class_filter(cn, checked)
            )
            self.class_filters_layout.addWidget(checkbox)
            self._selected_classes.add(class_name)
        
        # Update relay mapping table class dropdowns
        self._update_relay_mapping_classes()

    @Slot()
    def start_stream(self) -> None:
        if self._worker.isRunning():
            self.status_label.setText("Already running.")
            return
        index = self.camera_combo.currentData()
        if index is None:
            self.status_label.setText("Select a valid camera.")
            return
        self._worker.set_camera_index(int(index))
        # ensure worker uses selected compute device
        self._worker.set_device(self.compute_combo.currentText())
        self._worker.start()

    @Slot()
    def stop_stream(self) -> None:
        if self._worker.isRunning():
            self._worker.stop()
            self._worker.wait(2000)

    @Slot()
    def capture_frame(self) -> None:
        if self._current_pixmap is None:
            self.status_label.setText("No frame to capture.")
            return
        file_path, _ = QFileDialog.getSaveFileName(
            self,
            "Save Captured Frame",
            "",
            "Images (*.png *.jpg *.bmp)",
        )
        if not file_path:
            return
        if self._current_pixmap.save(file_path):
            self.status_label.setText(f"Frame saved: {Path(file_path).name}")
        else:
            self.status_label.setText("Frame save failed.")

    @Slot(int)
    def on_confidence_changed(self, value: int) -> None:
        self._confidence_threshold = value / 100.0
        self.confidence_label.setText(f"Confidence: {value}%")
        self._update_match_status_label()

    def _toggle_class_filter(self, class_name: str, checked: bool) -> None:
        if checked:
            self._selected_classes.add(class_name)
        else:
            self._selected_classes.discard(class_name)

    def _update_relay_mapping_classes(self) -> None:
        """Update all class dropdowns in the relay mapping table."""
        sorted_classes = sorted(self._model_classes)
        for row in range(8):
            combo = self.relay_mapping_table.cellWidget(row, 0)
            if combo is None:
                continue
            combo.blockSignals(True)
            current_data = combo.currentData()
            combo.clear()
            combo.addItem("None", None)
            for class_name in sorted_classes:
                combo.addItem(class_name, class_name)
            # Restore previous selection if still valid
            if current_data:
                idx = combo.findData(current_data)
                if idx >= 0:
                    combo.setCurrentIndex(idx)
            combo.blockSignals(False)
            # Sync mapping state
            self._relay_mappings[row]["class"] = combo.currentData()

    def _on_relay_mapping_class_changed(self, row: int) -> None:
        """Handle class dropdown change in relay mapping table."""
        combo = self.relay_mapping_table.cellWidget(row, 0)
        if combo is not None:
            self._relay_mappings[row]["class"] = combo.currentData()

    def _on_relay_mapping_channel_changed(self, row: int) -> None:
        """Handle channel dropdown change in relay mapping table."""
        combo = self.relay_mapping_table.cellWidget(row, 1)
        if combo is not None:
            self._relay_mappings[row]["channel"] = combo.currentData()

    def _update_match_status_label(self) -> None:
        """Update match status and relay state for all 8 mappings."""
        matched_classes = []
        for row, mapping in enumerate(self._relay_mappings):
            class_name = mapping["class"]
            channel = mapping["channel"]
            if class_name is None:
                # No class selected for this row
                if mapping["last_state"]:
                    self._set_relay_channel(channel, False)
                    mapping["last_state"] = False
                self._set_relay_status_cell(row, False)
                continue
            
            detected = class_name in self._class_counts and self._class_counts[class_name] > 0
            now = time.time()
            
            if detected:
                matched_classes.append(f"{class_name}→CH{channel}")
            
            if detected and not mapping["last_state"]:
                # Turn ON and record timestamp
                self._set_relay_channel(channel, True)
                mapping["last_state"] = True
                mapping["last_on_time"] = now
            elif not detected and mapping["last_state"]:
                # Only turn OFF if minimum hold time has elapsed
                elapsed_on = now - mapping["last_on_time"]
                if elapsed_on >= self._relay_min_on_seconds:
                    self._set_relay_channel(channel, False)
                    mapping["last_state"] = False
            
            self._set_relay_status_cell(row, mapping["last_state"])
        
        if matched_classes:
            self.match_status_label.setText(f"Matched: {', '.join(matched_classes)}")
        else:
            active = [m["class"] for m in self._relay_mappings if m["class"] is not None]
            if active:
                self.match_status_label.setText("Checking for matches...")
            else:
                self.match_status_label.setText("No class-to-relay mappings configured")

    def _set_relay_status_cell(self, row: int, is_on: bool) -> None:
        """Update the relay status cell in the mapping table."""
        item = self.relay_mapping_table.item(row, 2)
        if item is None:
            item = QTableWidgetItem()
            item.setTextAlignment(Qt.AlignCenter)
            self.relay_mapping_table.setItem(row, 2, item)
        if is_on:
            item.setText("ON")
            item.setForeground(QBrush(QColor("#51cf66")))
        else:
            item.setText("OFF")
            item.setForeground(QBrush(QColor("#ff6b6b")))

    def _set_relay_channel(self, channel: int, on: bool) -> None:
        """Send relay on/off command for a specific channel."""
        if not self._relay_connected or self._relay is None:
            return
        try:
            if on:
                self._relay.on(channel)
            else:
                self._relay.off(channel)
        except Exception as e:
            self.status_label.setText(f"Relay CH{channel} error: {str(e)}")

    @Slot(str)
    def on_compute_changed(self, device: str) -> None:
        # Update status and tell worker to use selected device
        self.status_label.setText(f"Compute: {device}")
        self._worker.set_device(device)

    @Slot(float)
    def on_zoom_changed(self, zoom_level: float) -> None:
        """Handle zoom level changes."""
        self._zoom_level = zoom_level
        self.zoom_label.setText(f"Zoom: {zoom_level * 100:.0f}%")

    @Slot()
    def zoom_in(self) -> None:
        """Zoom in by 10%."""
        self.video_label.set_zoom_level(self._zoom_level * 1.1)

    @Slot()
    def zoom_out(self) -> None:
        """Zoom out by 10%."""
        self.video_label.set_zoom_level(self._zoom_level / 1.1)

    @Slot()
    def reset_zoom(self) -> None:
        """Reset zoom to 100%."""
        self.video_label.set_zoom_level(1.0)

    @Slot()
    def connect_relay(self) -> None:
        """Connect to relay board."""
        if not RELAY_AVAILABLE:
            self.relay_status_label.setText("Status: Relay library not available")
            self.relay_status_label.setStyleSheet("font-size: 11pt; color: #ff6b6b; font-weight: bold;")
            return
        
        try:
            self._relay_host = self.relay_host_input.text().strip()
            self._relay_port = self.relay_port_spin.value()
            
            if self._relay is not None:
                try:
                    self._relay.disconnect()
                except Exception:
                    pass
            
            self._relay = Relay(host=self._relay_host, port=self._relay_port)
            self._relay.connect()
            self._relay_connected = True
            
            self.relay_status_label.setText(f"Status: Connected ({self._relay_host}:{self._relay_port})")
            self.relay_status_label.setStyleSheet("font-size: 11pt; color: #51cf66; font-weight: bold;")
            self.relay_connect_button.setText("🔌 Disconnect Relay")
            self.relay_connect_button.clicked.disconnect()
            self.relay_connect_button.clicked.connect(self.disconnect_relay)
            
            self.status_label.setText(f"Relay connected to {self._relay_host}:{self._relay_port}")
        except Exception as e:
            self.relay_status_label.setText("Status: Connection failed")
            self.relay_status_label.setStyleSheet("font-size: 11pt; color: #ff6b6b; font-weight: bold;")
            self.status_label.setText(f"Relay connection error: {str(e)}")
            self._relay_connected = False

    @Slot()
    def disconnect_relay(self) -> None:
        """Disconnect from relay board and turn off all active channels."""
        try:
            if self._relay is not None and self._relay_connected:
                for mapping in self._relay_mappings:
                    if mapping["last_state"]:
                        try:
                            self._relay.off(mapping["channel"])
                        except Exception:
                            pass
                        mapping["last_state"] = False
                        mapping["last_on_time"] = 0.0
                self._relay.disconnect()
            self._relay_connected = False
            self._relay = None
            
            # Reset all status cells
            for row in range(8):
                self._set_relay_status_cell(row, False)
            
            self.relay_status_label.setText("Status: Disconnected")
            self.relay_status_label.setStyleSheet("font-size: 11pt; color: #ff6b6b; font-weight: bold;")
            self.relay_connect_button.setText("🔌 Connect Relay")
            self.relay_connect_button.clicked.disconnect()
            self.relay_connect_button.clicked.connect(self.connect_relay)
            
            self.status_label.setText("Relay disconnected")
        except Exception as e:
            self.status_label.setText(f"Relay disconnection error: {str(e)}")

    @Slot(QImage, list)
    def on_frame(self, image: QImage, detections: list) -> None:
        # Update FPS counter
        current_time = time.time()
        if self._fps_timer == 0:
            self._fps_timer = current_time
        
        self._fps_counter += 1
        elapsed = current_time - self._fps_timer
        if elapsed >= 1.0:
            self._current_fps = self._fps_counter / elapsed
            self.fps_label.setText(f"FPS: {self._current_fps:.1f}")
            self._fps_counter = 0
            self._fps_timer = current_time
        
        pixmap = QPixmap.fromImage(image)
        
        # Filter detections based on selected classes and confidence threshold
        filtered_detections = []
        for det in detections:
            class_name = det.get("class_name", "unknown")
            conf = float(det["label"].split()[-1]) if " " in det["label"] else 0.0
            
            # Filter by selected classes and confidence
            if class_name in self._selected_classes and conf >= self._confidence_threshold:
                filtered_detections.append(det)
        
        detections = filtered_detections  # Use filtered for drawing and table
        
        if detections:
            # Assign colors to each detection based on class name
            for det in detections:
                class_name = det.get("class_name", "unknown")
                if class_name not in self._class_colors:
                    hue = (len(self._class_colors) * 37) % 360
                    self._class_colors[class_name] = QColor.fromHsv(hue, 200, 255)
                det["color"] = self._class_colors[class_name]
            
            painter = QPainter(pixmap)
            pen = QPen()
            pen.setWidth(2)
            for det in detections:
                color = det.get("color", Qt.green)
                pen.setColor(color)
                painter.setPen(pen)
                x1, y1, x2, y2 = det["x1"], det["y1"], det["x2"], det["y2"]
                painter.drawRect(x1, y1, x2 - x1, y2 - y1)
                painter.drawText(x1, max(0, y1 - 6), det["label"])
            painter.end()
        
        # Update total detections
        self._total_detections = len(detections)
        self.detections_label.setText(f"Detections: {self._total_detections}")
        
        self._current_pixmap = pixmap
        # Set pixmap to zoomable label (it will handle scaling with zoom level)
        self.video_label.setPixmap(pixmap)
        self._update_class_table(detections)

    def _update_class_table(self, detections: list) -> None:
        self._class_counts.clear()
        for det in detections:
            class_name = det.get("class_name", "unknown")
            self._class_counts[class_name] = self._class_counts.get(class_name, 0) + 1

        self.table.setRowCount(len(self._class_counts))
        for row, (class_name, count) in enumerate(sorted(self._class_counts.items())):
            color = self._class_colors.get(class_name, Qt.white)

            class_item = QTableWidgetItem(class_name)
            count_item = QTableWidgetItem(str(count))
            brush = QBrush(color)
            class_item.setForeground(brush)
            count_item.setForeground(brush)

            self.table.setItem(row, 0, class_item)
            self.table.setItem(row, 1, count_item)
        
        # Update match status label based on actual detections
        self._update_match_status_label()

    @Slot(str)
    def on_status(self, message: str) -> None:
        self.status_label.setText(message)

    def closeEvent(self, event) -> None:
        self.stop_stream()
        self.disconnect_relay()
        super().closeEvent(event)


def main() -> None:
    app = QApplication(sys.argv)
    app.setStyle("Fusion")
    
    window = MainWindow()
    window.resize(1400, 800)
    window.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
