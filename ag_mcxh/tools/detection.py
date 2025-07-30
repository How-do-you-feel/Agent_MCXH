from .base import BaseTool
from .registry import register_tool
from typing import Any
import os

@register_tool("YoloDetect")
class YoloDetect(BaseTool):
    
    @property
    def default_desc(self) -> str:
        return "Detect objects in images using YOLO model"
    
    def __init__(self, 
                 model_name: str = "YOLO",
                 model_path: str = "yolo11n.pt",
                 device: str = "cpu",
                 conf_threshold: float = 0.5,
                 **kwargs):
        super().__init__(**kwargs)
        self.model_name = model_name
        self.model_path = model_path
        self.device = device
        self.conf_threshold = conf_threshold
        self._model = None
        
    def setup(self):
        try:
            from ..models.registry import load_model
            model_cls = load_model(self.model_name)
            self._model = model_cls(self.model_path)
            self._model.to(self.device)
        except Exception as e:
            raise RuntimeError(f"Failed to load model {self.model_name}: {str(e)}")
    
    def apply(self, image_path: str) -> str:
        if self._model is None:
            self.setup()
            
        if not os.path.exists(image_path):
            return f"Error: Image file not found {image_path}"
            
        try:
            results = self._model(image_path, conf=self.conf_threshold)
            result = results[0]
            
            detections = []
            if result.boxes is not None:
                for box in result.boxes:
                    detection = {
                        'class_id': int(box.cls.item()),
                        'class_name': result.names[int(box.cls.item())],
                        'confidence': float(box.conf.item()),
                        'bbox': {
                            'x1': int(box.xyxy[0][0].item()),
                            'y1': int(box.xyxy[0][1].item()),
                            'x2': int(box.xyxy[0][2].item()),
                            'y2': int(box.xyxy[0][3].item())
                        }
                    }
                    detections.append(detection)
            
            return str(detections)
        except Exception as e:
            return f"Detection failed: {str(e)}"