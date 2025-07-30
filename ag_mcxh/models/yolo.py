from .registry import register_model
from typing import Any

@register_model("YOLOv8")
class YOLOv8:
    """YOLOv8 Model"""
    
    def __init__(self, model_path: str = "yolov8n.pt", device: str = "cpu"):
        self.model_path = model_path
        self.device = device
        self._model = None
    
    def load(self):
        """Load the model"""
        if self._model is None:
            try:
                from ultralytics import YOLO
                self._model = YOLO(self.model_path)
                self._model.to(self.device)
            except ImportError:
                raise ImportError("Please install ultralytics: pip install ultralytics")
        return self._model
    
    def predict(self, image_path: str, **kwargs):
        """Run prediction on image"""
        model = self.load()
        return model(image_path, **kwargs)

@register_model("YOLOv5")
class YOLOv5:
    """YOLOv5 Model"""
    
    def __init__(self, model_path: str = "yolov5s.pt", device: str = "cpu"):
        self.model_path = model_path
        self.device = device
        self._model = None
    
    def load(self):
        """Load the model"""
        if self._model is None:
            try:
                import torch
                self._model = torch.hub.load('ultralytics/yolov5', 'custom', path=self.model_path)
                self._model.to(self.device)
            except ImportError:
                raise ImportError("Please install torch and yolov5")
        return self._model
    
    def predict(self, image_path: str, **kwargs):
        """Run prediction on image"""
        model = self.load()
        return model(image_path, **kwargs)