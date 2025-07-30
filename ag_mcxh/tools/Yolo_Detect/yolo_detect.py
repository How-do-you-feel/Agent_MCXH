from ...tools.base import BaseTool
from ...types import Annotated, ImageIO, Info
import sys
import os
import argparse
import torch
from ultralytics import YOLO
import cv2
import numpy as np
import json

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_path', type=str, 
                       default='./yolo11n.pt')
    parser.add_argument('--image_path', type=str, 
                       default='/home/ps/MCXH/Agent_MCXH/pics/002.png')
    parser.add_argument('--device', type=str, default='cuda')
    parser.add_argument('--conf_threshold', type=float, default=0.5)
    parser.add_argument('--iou_threshold', type=float, default=0.45)
    parser.add_argument('--image_size', type=int, default=640)
    parser.add_argument('--classes', type=str, default='')
    parser.add_argument('--output_path', type=str, 
                       default='./output/result.jpg')
    return parser.parse_args()

def run_yolo_detection(image_path, model_path, device, conf_threshold, iou_threshold, image_size, classes, output_path):
    model = YOLO(model_path)
    model.to(device)
    
    image = cv2.imread(image_path)
    if image is None:
        raise ValueError(f"Could not load image from {image_path}")

    class_indices = None
    if classes:
        try:
            class_indices = [int(x) for x in classes.split(',')]
        except ValueError:
            print("Invalid class indices, using all classes")
    
    results = model(
        image,
        conf=conf_threshold,
        iou=iou_threshold,
        imgsz=image_size,
        device=device,
        classes=class_indices
    )
    
    result = results[0]
    
    annotated_image = result.plot()

    output_dir = os.path.dirname(output_path)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    cv2.imwrite(output_path, annotated_image)

    detections = []
    for box in result.boxes:
        detection = {
            'class_id': int(box.cls.item()),
            'class_name': model.names[int(box.cls.item())],
            'confidence': float(box.conf.item()),
            'bbox': {
                'x1': int(box.xyxy[0][0].item()),
                'y1': int(box.xyxy[0][1].item()),
                'x2': int(box.xyxy[0][2].item()),
                'y2': int(box.xyxy[0][3].item())
            }
        }
        detections.append(detection)
    
    output = {
        'detections': detections,
        'image_path': output_path
    }
    
    return json.dumps(output)

class YoloDetect(BaseTool):
    """A tool to detect objects using YOLO model."""
    
    default_desc = ('This tool can detect objects in images using YOLO model '
                    'and return detection results with bounding boxes.')
    
    def __init__(self, 
                 model_path: str = '',
                 device: str = 'cuda',
                 conf_threshold: float = 0.5,
                 iou_threshold: float = 0.45,
                 image_size: int = 640,
                 classes: str = '',
                 output_path: str = '',
                 toolmeta=None):
        super().__init__(toolmeta=toolmeta)
        self.model_path = model_path
        self.device = device
        self.conf_threshold = conf_threshold
        self.iou_threshold = iou_threshold
        self.image_size = image_size
        self.classes = classes
        self.output_path = output_path
        self.model = YOLO(self.model_path)
        self.model.to(self.device)
        
    def apply(self, image: ImageIO) -> Annotated[str, Info('Detection results with object names, confidence and bounding boxes.')]:
        """Apply YOLO detection to the image.
        
        Args:
            image (ImageIO): Input image
            
        Returns:
            str: Detection results in JSON format with object names, confidence and bounding boxes
        """
        image_path = image.image_path
        result = run_yolo_detection(
            image_path=image_path,
            model_path=self.model_path,
            device=self.device,
            conf_threshold=self.conf_threshold,
            iou_threshold=self.iou_threshold,
            image_size=self.image_size,
            classes=self.classes,
            output_path=self.output_path
        )
        
        return result

def main():
    args = parse_args()
    
    result = run_yolo_detection(
        image_path=args.image_path,
        model_path=args.model_path,
        device=args.device,
        conf_threshold=args.conf_threshold,
        iou_threshold=args.iou_threshold,
        image_size=args.image_size,
        classes=args.classes,
        output_path=args.output_path
    )
    
    print(result)

if __name__ == '__main__':
    main()