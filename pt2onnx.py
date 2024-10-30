import argparse
import cv2
import numpy as np
import onnxruntime as ort
import torch

import sys
sys.path.insert(0, './ultralytics-main/')
from ultralytics import YOLO
from ultralytics.utils.checks import check_yaml
from ultralytics.utils import yaml_load

class YOLOv8:
    def __init__(self, onnx_model, confidence_thres, iou_thres):
        """
        初始化YOLOv8类的实例。
        参数:
            onnx_model: ONNX模型的路径。
            input_image: 输入图像的路径。
            confidence_thres: 过滤检测的置信度阈值。
            iou_thres: 非极大抑制的IoU（交并比）阈值。
        """
        self.onnx_model = onnx_model

        self.confidence_thres = confidence_thres
        self.iou_thres = iou_thres

        # 从COCO数据集的配置文件加载类别名称
        self.classes = yaml_load(check_yaml("../yolo_analysis/coco8.yaml"))["names"]
        # 字典存储类别名称
        print(self.classes)
        # {0: 'person', 1: 'bicycle', 2: 'car', 3: 'motorcycle', 4: 'airplane'...}

        # 为类别生成颜色调色板
        self.color_palette = np.random.uniform(0, 255, size=(len(self.classes), 3))

        # 初始化ONNX会话
        self.initialize_session(self.onnx_model)
