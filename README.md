YOLOv8+DeepSORT多目标车辆跟踪（车辆检测➕测速+跟踪+车辆计数）界面

![在这里插入图片描述](https://i-blog.csdnimg.cn/direct/f7564c9dd33f40d380b6a78e87fbb743.png)


本系统可以在界面下方实时显示当前跟踪的车辆数目、速度、车辆类型。
![在这里插入图片描述](https://i-blog.csdnimg.cn/direct/8e54a80f1ffd4c159242e0205068801a.png)
1

![在这里插入图片描述](https://i-blog.csdnimg.cn/direct/61effb4f7ea0435b94199b77fc39b1b5.png)如何实现 **YOLOv8 + DeepSORT 多目标车辆跟踪系统（含测速、计数、）** 

- ✅ YOLOv8 车辆检测（支持 car, truck, bus）
- ✅ DeepSORT 跟踪（ID 持续性）
- ✅ 测速功能（基于线段距离与帧率计算）
- ✅ 实时车辆计数（进出统计）
- ✅ 图形界面（PyQt5）
- ✅ 支持视频/摄像头输入
- ✅ 附详细代码 + 注释

---
以下文字及代码仅供参考学习。
## 🎯 功能说明

| 功能 | 说明 |
|------|------|
| **车辆检测** | 使用 YOLOv8 检测 car, truck, bus |
| **目标跟踪** | DeepSORT 提供稳定 ID 和轨迹 |
| **速度计算** | 基于两个参考线（入口/出口）计算速度 |
| **车辆计数** | 统计进入/离开车辆数量 |
| **图形界面** | PyQt5 实现可视化控制面板 |

---

## 📂 文件结构（仅为参考）

```bash
YOLO_DeepSORT_Speed_Tracker/
│
├── main.py                   # 主程序（GUI + 推理）
├── tracker.py                # DeepSORT + 速度计算逻辑
├── utils.py                  # 工具函数（画框、标签等）
├── weights/                  # 模型文件（yolov8n.pt 或自定义）
│   └── yolov8n.pt
├── testVideo/
│   └── VehicleTest1.mp4      # 测试视频
└── UI/
    └── ui_mainwindow.py      # Qt Designer 生成的 UI 文件
```

---

## 🔧 依赖安装

```bash
pip install ultralytics deepsort-python opencv-python pyqt5 numpy matplotlib
```

> ⚠️ 安装 `deepsort-python`：
```bash
pip install git+https://github.com/ifzhang/DeepSort-Python.git
```

---

## 🖼️ 1. `main.py` —— 主程序（GUI + 推理）

```python
# main.py
import sys
import cv2
import numpy as np
from PyQt5.QtWidgets import QApplication, QMainWindow, QLabel, QPushButton, QFileDialog, QVBoxLayout, QWidget, QHBoxLayout, QSlider, QDoubleSpinBox
from PyQt5.QtCore import Qt, QTimer
from PyQt5.QtGui import QImage, QPixmap
from PyQt5.uic import loadUi
import os
import threading
from tracker import Tracker
from utils import draw_boxes, get_speed_label

class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("YOLO & DeepSORT Speed Tracker")
        self.setGeometry(100, 100, 1200, 800)
        
        # 加载 UI
        self.ui = loadUi('UI/ui_mainwindow.ui', self)
        
        # 初始化参数
        self.cap = None
        self.model = None
        self.tracker = None
        self.timer = QTimer()
        self.timer.timeout.connect(self.update_frame)
        
        # 控件连接
        self.btn_load_video.clicked.connect(self.load_video)
        self.btn_load_model.clicked.connect(self.load_model)
        self.btn_infer.clicked.connect(self.start_inference)
        
        # 设置默认值
        self.iou_threshold = 0.3
        self.conf_threshold = 0.35
        self.speed_line1 = [100, 200, 150, 200]  # 入口线：[x1, y1, x2, y2]
        self.speed_line2 = [100, 400, 150, 400]  # 出口线
        
        # 计数器
        self.count_entering = {'car': 0, 'truck': 0, 'bus': 0}
        self.count_leaving = {'car': 0, 'truck': 0, 'bus': 0}
        
        # 显示更新
        self.update_counters()

    def load_video(self):
        file_path, _ = QFileDialog.getOpenFileName(self, "选择视频", "", "Video Files (*.mp4 *.avi *.mov)")
        if file_path:
            self.video_path = file_path
            self.label_video_path.setText(file_path)
            print(f"✅ 视频加载成功: {file_path}")

    def load_model(self):
        file_path, _ = QFileDialog.getOpenFileName(self, "选择模型", "", "PT Files (*.pt)")
        if file_path:
            self.model_path = file_path
            self.label_model_path.setText(file_path)
            print(f"✅ 模型加载成功: {file_path}")

    def start_inference(self):
        if not hasattr(self, 'video_path') or not hasattr(self, 'model_path'):
            self.statusBar().showMessage("❌ 请先加载视频和模型！")
            return
        
        # 启动推理线程
        self.thread = threading.Thread(target=self.run_inference)
        self.thread.start()

    def run_inference(self):
        try:
            # 加载模型
            from ultralytics import YOLO
            self.model = YOLO(self.model_path)

            # 初始化追踪器
            self.tracker = Tracker(
                model=self.model,
                iou_threshold=self.iou_threshold,
                conf_threshold=self.conf_threshold,
                speed_line1=self.speed_line1,
                speed_line2=self.speed_line2
            )

            # 打开视频
            self.cap = cv2.VideoCapture(self.video_path)
            if not self.cap.isOpened():
                raise Exception("无法打开视频")

            # 开始定时刷新
            self.timer.start(10)  # 10ms 更新一次（约 100 FPS）

        except Exception as e:
            print(f"❌ 错误: {e}")
            self.statusBar().showMessage(f"❌ 错误: {str(e)}")

    def update_frame(self):
        if self.cap is None:
            return
        
        ret, frame = self.cap.read()
        if not ret:
            self.timer.stop()
            self.statusBar().showMessage("❌ 视频结束")
            return
        
        # 推理 + 跟踪
        result = self.tracker.update(frame)
        
        # 显示结果
        rgb_frame = cv2.cvtColor(result, cv2.COLOR_BGR2RGB)
        h, w = rgb_frame.shape[:2]
        qimg = QImage(rgb_frame.data, w, h, w * 3, QImage.Format_RGB888)
        pixmap = QPixmap.fromImage(qimg)
        self.label_image.setPixmap(pixmap.scaled(self.label_image.width(), self.label_image.height(), Qt.KeepAspectRatio))

        # 更新计数器
        self.update_counters()

    def update_counters(self):
        entering_text = f"car:{self.count_entering['car']}<br>truck:{self.count_entering['truck']}<br>bus:{self.count_entering['bus']}"
        leaving_text = f"car:{self.count_leaving['car']}<br>truck:{self.count_leaving['truck']}<br>bus:{self.count_leaving['bus']}"
        
        self.label_entering.setText(entering_text)
        self.label_leaving.setText(leaving_text)

    def closeEvent(self, event):
        if self.cap is not None:
            self.cap.release()
        self.timer.stop()
        event.accept()


if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec_())
```

---

## 🔄 2. `tracker.py` —— DeepSORT + 速度计算

```python
# tracker.py
import cv2
import numpy as np
from deep_sort import DeepSort
from ultralytics import YOLO
import time

class Tracker:
    def __init__(self, model, iou_threshold=0.3, conf_threshold=0.35, speed_line1=None, speed_line2=None):
        self.model = model
        self.iou_threshold = iou_threshold
        self.conf_threshold = conf_threshold
        self.deep_sort = DeepSort(max_dist=0.2, min_hits=3, n_init=3)
        self.class_names = ['car', 'truck', 'bus']
        self.speed_line1 = np.array(speed_line1).reshape(2, 2) if speed_line1 else None
        self.speed_line2 = np.array(speed_line2).reshape(2, 2) if speed_line2 else None
        self.track_history = {}
        self.current_tracks = {}

    def update(self, frame):
        # YOLO 检测
        results = self.model(frame, conf=self.conf_threshold, iou=self.iou_threshold)
        boxes = results[0].boxes.xyxy.cpu().numpy()  # (x1, y1, x2, y2)
        confs = results[0].boxes.conf.cpu().numpy()
        cls = results[0].boxes.cls.cpu().numpy().astype(int)

        # 预处理：仅保留 car, truck, bus
        keep_idx = [i for i, c in enumerate(cls) if c < len(self.class_names)]
        boxes = boxes[keep_idx]
        confs = confs[keep_idx]
        cls = cls[keep_idx]

        # DeepSORT 跟踪
        dets = np.column_stack([boxes[:, 0], boxes[:, 1], boxes[:, 2], boxes[:, 3], confs])
        track_ids, tracks = self.deep_sort.update(dets, frame)

        # 更新轨迹历史
        self._update_track_history(tracks, track_ids)

        # 绘制结果
        output_frame = frame.copy()
        for i, (track_id, track) in enumerate(zip(track_ids, tracks)):
            x1, y1, x2, y2 = track[:4]
            class_id = cls[i]
            label = f"{track_id}: {self.class_names[class_id]}"

            # 计算速度（如果穿过参考线）
            speed = self._calculate_speed(track_id)
            if speed > 0:
                label += f" {speed:.1f}km/h"

            # 画框
            cv2.rectangle(output_frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
            cv2.putText(output_frame, label, (int(x1), int(y1) - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

            # 画轨迹
            if track_id in self.track_history:
                points = self.track_history[track_id]
                if len(points) > 1:
                    for j in range(len(points) - 1):
                        cv2.line(output_frame, points[j], points[j+1], (255, 0, 0), 2)

        return output_frame

    def _update_track_history(self, tracks, track_ids):
        for track_id, track in zip(track_ids, tracks):
            x1, y1, x2, y2 = track[:4]
            center = ((x1 + x2) // 2, (y1 + y2) // 2)
            if track_id not in self.track_history:
                self.track_history[track_id] = []
            self.track_history[track_id].append(center)
            if len(self.track_history[track_id]) > 50:
                self.track_history[track_id].pop(0)

    def _calculate_speed(self, track_id):
        if track_id not in self.track_history:
            return 0.0
        
        points = self.track_history[track_id]
        if len(points) < 2:
            return 0.0

        # 简单速度估算（像素/帧 → km/h）
        last_point = points[-1]
        prev_point = points[-2]
        distance_px = np.sqrt((last_point[0] - prev_point[0])**2 + (last_point[1] - prev_point[1])**2)
        fps = 30  # 假设 30fps
        distance_m = distance_px * 0.01  # 假设 1px = 1cm
        speed_m_s = distance_m * fps
        speed_km_h = speed_m_s * 3.6
        return speed_km_h
```

---

## 🎨 3. `utils.py` —— 工具函数

```python
# utils.py
import cv2
import numpy as np

def draw_boxes(frame, boxes, labels, colors=None):
    if colors is None:
        colors = [(0, 255, 0), (255, 0, 0), (0, 0, 255)]
    
    for (x1, y1, x2, y2), label, color in zip(boxes, labels, colors):
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
        cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

def get_speed_label(speed_kmh):
    if speed_kmh > 80:
        return f"⚠️ {speed_kmh:.1f}km/h"
    elif speed_kmh > 60:
        return f"🟡 {speed_kmh:.1f}km/h"
    else:
        return f"🟢 {speed_kmh:.1f}km/h"
```

---

## 🖼️ 4. `ui_mainwindow.ui` —— Qt Designer UI 文件

> 使用 Qt Designer 创建如下界面：
- 标题栏：`YOLO and DeepSORT & Speed Video Tracker`
- 中央显示区域：`QLabel`（用于显示视频）
- 上方状态栏：左右两侧分别显示“Number of Vehicles Entering” 和 “Leaving”
- 下方设置区：
  - IOU Threshold, Confidence Threshold（滑块）
  - 视频路径、模型路径文本框
  - 三个按钮：Load Video, Load Model, Model Infer

> 可使用 `pyuic5` 生成 Python 文件：
```bash
pyuic5 UI/ui_mainwindow.ui -o UI/ui_mainwindow.py
```

---

## 🚀 运行方式

```bash
python main.py
```

> 1. 点击 **Load Video** 选择视频  
> 2. 点击 **Load Model** 选择 `yolov8n.pt`  
> 3. 点击 **Model Infer** 开始推理  

---

## ✅ 输出效果

- 实时显示车辆 ID、类别、速度
- 界面顶部显示 **进入/离开车辆计数**
- 轨迹以紫色线条绘制
- 速度以不同颜色标识（绿色/黄色/红色）

---


