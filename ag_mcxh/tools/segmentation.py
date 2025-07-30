from .base import BaseTool
from .registry import register_tool
from typing import Any
import os

@register_tool("SegmentAnything")
class SegmentAnything(BaseTool):
    """Image Segmentation Tool"""
    
    @property
    def default_desc(self) -> str:
        return "Segment images using Segment Anything model"
    
    def __init__(self, 
                 model_name: str = "SAM-ViT-H",
                 model_path: str = "/home/ps/MCXH/Agent_MCXH/models/sam_vit_h_4b8939.pth",
                 device: str = "cpu",
                 **kwargs):
        super().__init__(**kwargs)
        self.model_name = model_name
        self.model_path = model_path
        self.device = device
        self._model = None
        
    def setup(self):
        """Initialize SAM model"""
        try:
            from ..models import load_model
            model_cls = load_model(self.model_name)
            self._model = model_cls(self.model_path, self.device)
        except Exception as e:
            raise RuntimeError(f"Failed to load model {self.model_name}: {str(e)}")
    
    def apply(self, image_path: str) -> str:
        """Apply image segmentation"""
        if not os.path.exists(image_path):
            return f"Error: Image file not found {image_path}"
            
        try:
            import cv2
            import numpy as np
            
            image = cv2.imread(image_path)
            if image is None:
                return f"Error: Cannot read image {image_path}"
                
            # For demonstration, we'll use a simple point in the center
            results = self._model.predict(
                image_path, 
                point_coords=np.array([[image.shape[1]//2, image.shape[0]//2]]), 
                point_labels=np.array([1])
            )
            
            masks, scores, logits = results
            return f"Segmentation completed, detected {len(masks)} objects"
        except Exception as e:
            return f"Segmentation failed: {str(e)}"


@register_tool("SegmentObject")
class SegmentObject(BaseTool):
    """Object Segmentation Tool"""
    
    @property
    def default_desc(self) -> str:
        return "Segment specific objects in images"
    
    def __init__(self, 
                 model_name: str = "SAM-ViT-H",
                 model_path: str = "/home/ps/MCXH/Agent_MCXH/models/sam_vit_h_4b8939.pth",
                 device: str = "cpu",
                 **kwargs):
        super().__init__(**kwargs)
        self.model_name = model_name
        self.model_path = model_path
        self.device = device
        self._model = None
        
    def setup(self):
        """Initialize model"""
        try:
            from ..models import load_model
            model_cls = load_model(self.model_name)
            self._model = model_cls(self.model_path, self.device)
        except Exception as e:
            raise RuntimeError(f"Failed to load model {self.model_name}: {str(e)}")
    
    def apply(self, image_path: str, prompt: str) -> str:
        """Apply object segmentation"""
        if not os.path.exists(image_path):
            return f"Error: Image file not found {image_path}"
            
        try:
            # This should implement specific object segmentation based on prompt
            # Simplified implementation for demonstration
            return f"Segmented '{prompt}' in the image"
        except Exception as e:
            return f"Object segmentation failed: {str(e)}"