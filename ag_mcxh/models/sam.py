from .registry import register_model
from typing import Any

@register_model("SAM-ViT-H")
class SAMViTH:
    """SAM ViT-H Model"""
    
    def __init__(self, model_path: str = "sam_vit_h_4b8939.pth", device: str = "cpu"):
        self.model_path = model_path
        self.device = device
        self._predictor = None
    
    def load(self):
        """Load the model"""
        if self._predictor is None:
            try:
                from segment_anything import sam_model_registry, SamPredictor
                import torch
                
                sam = sam_model_registry["vit_h"](checkpoint=self.model_path)
                sam.to(device=self.device)
                self._predictor = SamPredictor(sam)
            except ImportError:
                raise ImportError("Please install segment-anything: pip install git+https://github.com/facebookresearch/segment-anything.git")
        return self._predictor
    
    def predict(self, image_path: str, **kwargs):
        """Run prediction on image"""
        import cv2
        predictor = self.load()
        
        image = cv2.imread(image_path)
        if image is None:
            raise ValueError(f"Cannot read image: {image_path}")
            
        predictor.set_image(image)
        return predictor.predict(**kwargs)

@register_model("SAM-ViT-L")
class SAMViTL:
    """SAM ViT-L Model"""
    
    def __init__(self, model_path: str = "sam_vit_l_0b3195.pth", device: str = "cpu"):
        self.model_path = model_path
        self.device = device
        self._predictor = None
    
    def load(self):
        """Load the model"""
        if self._predictor is None:
            try:
                from segment_anything import sam_model_registry, SamPredictor
                import torch
                
                sam = sam_model_registry["vit_l"](checkpoint=self.model_path)
                sam.to(device=self.device)
                self._predictor = SamPredictor(sam)
            except ImportError:
                raise ImportError("Please install segment-anything: pip install git+https://github.com/facebookresearch/segment-anything.git")
        return self._predictor
    
    def predict(self, image_path: str, **kwargs):
        """Run prediction on image"""
        import cv2
        predictor = self.load()
        
        image = cv2.imread(image_path)
        if image is None:
            raise ValueError(f"Cannot read image: {image_path}")
            
        predictor.set_image(image)
        return predictor.predict(**kwargs)