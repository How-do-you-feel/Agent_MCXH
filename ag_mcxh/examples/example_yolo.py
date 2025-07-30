from ag_mcxh.apis import load_tool
from ag_mcxh.types import ImageIO

yolo_tool = load_tool('YoloDetect',
                      model_path='/home/ps/MCXH/MingChaXinHao/ag_mcxh/ag_mcxh/tools/Yolo_Detect/yolo11n.pt',
                      device='cpu',  
                      conf_threshold=0.5,
                      iou_threshold=0.45,
                      image_size=640)

image = ImageIO('/home/ps/MCXH/MingChaXinHao/ag_mcxh/pics/002.png')

detection_results = yolo_tool.apply(image)
print(detection_results)