import sys
import time
from pathlib import Path
from typing import List, Optional

import cv2
import numpy as np
from PySide6.QtCore import Qt, QThread, Signal, Slot
from PySide6.QtGui import QBrush, QColor, QFont, QImage, QPainter, QPen, QPixmap
from PySide6.QtWidgets import (
    QApplication,
    QCheckBox,
    QComboBox,
    QFileDialog,
    QFrame,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QScrollArea,
    QSlider,
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
    QWidget,
)

try:
    from ultralytics import YOLO  # type: ignore
except Exception:
    YOLO = None


ALLOWED_FPS = [24, 30, 60]


def find_cameras(max_index: int = 10) -> List[int]:
    available = []
    for idx in range(max_index):
        cap = cv2.VideoCapture(idx, cv2.CAP_DSHOW)
        if cap.isOpened():
            available.append(idx)
            cap.release()
        else:
            cap.release()
    return available


def nearest_allowed_fps(value: float) -> int:
    if value <= 1:
        return 30
    return min(ALLOWED_FPS, key=lambda x: abs(x - value))


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

    def set_camera_index(self, index: Optional[int]) -> None:
        self._camera_index = index

    def set_model_path(self, path: Optional[Path]) -> None:
        self._model_path = path
        self._model = None

    def stop(self) -> None:
        self._running = False

    def _load_model(self) -> None:
        if self._model_path is None:
            return
        if YOLO is None:
            self.status.emit("Ultralytics not available. Install requirements.")
            return
        try:
            self._model = YOLO(str(self._model_path))
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

        cap = cv2.VideoCapture(self._camera_index, cv2.CAP_DSHOW)
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
        
        header_layout.addWidget(QLabel("Model:"), 0)
        header_layout.addWidget(self.model_label, 1)
        header_layout.addWidget(self.load_model_button, 0)
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
        
        # Video Display
        self.video_label = QLabel()
        self.video_label.setAlignment(Qt.AlignCenter)
        self.video_label.setMinimumSize(640, 480)
        self.video_label.setStyleSheet("background-color: #000000; border-radius: 8px;")
        
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
        
        filter_layout.addSpacing(15)
        
        # Match Alert Section
        match_label = QLabel("🎯 Match Alert")
        match_label.setStyleSheet("font-size: 10pt; font-weight: bold;")
        filter_layout.addWidget(match_label)
        
        self.match_combo = QComboBox()
        self.match_combo.addItem("None", None)
        self.match_combo.currentIndexChanged.connect(self.on_match_class_changed)
        filter_layout.addWidget(self.match_combo)
        
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
        
        # Sidebar with filter and table
        # Main content area
        content_layout = QHBoxLayout()
        content_layout.addWidget(self.video_label, 2)
        content_layout.addWidget(filter_frame, 0)
        content_layout.addWidget(table_frame, 0)
        
        # Main layout
        layout = QVBoxLayout()
        layout.addWidget(header_frame, 0)
        layout.addWidget(metrics_frame, 0)
        layout.addWidget(self.match_status_label, 0)
        layout.addWidget(cam_frame, 0)
        layout.addLayout(content_layout, 1)
        
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
        
        # Update match combo dropdown
        self.match_combo.blockSignals(True)
        self.match_combo.clear()
        self.match_combo.addItem("None", None)
        for class_name in sorted(self._model_classes):
            self.match_combo.addItem(class_name, class_name)
        self.match_combo.blockSignals(False)

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

    def _update_match_status_label(self) -> None:
        """Update match status label based on detected classes in table."""
        if self._match_class:
            # Check if match class is in current detections
            if self._match_class in self._class_counts and self._class_counts[self._match_class] > 0:
                self.match_status_label.setText(
                    f"Matched = {self._match_class} ({self._class_counts[self._match_class]} detected)"
                )
            else:
                self.match_status_label.setText(
                    "Checking for matches..."
                )
        else:
            self.match_status_label.setText(
                "Checking for matches..."
            )

    @Slot()
    def on_match_class_changed(self) -> None:
        self._match_class = self.match_combo.currentData()
        self._match_detected = False
        self._update_match_status_label()

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
        self._match_detected = False
        for det in detections:
            class_name = det.get("class_name", "unknown")
            conf = float(det["label"].split()[-1]) if " " in det["label"] else 0.0
            
            # Check if this detection matches the alert class
            if self._match_class and class_name == self._match_class:
                self._match_detected = True
            
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
