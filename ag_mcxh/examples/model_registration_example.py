import sys
import os

project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ag_mcxh_path = os.path.join(project_root, 'ag_mcxh')

if project_root not in sys.path:
    sys.path.insert(0, project_root)
if ag_mcxh_path not in sys.path:
    sys.path.insert(0, ag_mcxh_path)

from ag_mcxh.models import list_models, load_model
from ag_mcxh import load_tool, list_tools

def main():
    print("Model Registration Example")
    print("=" * 30)
    print("Registered Models:")
    models = list_models()
    for model in models:
        print(f"  - {model}")
    
    print("\n" + "-" * 30)
    
    print("Registered Tools:")
    tools = list_tools()
    for tool in tools:
        print(f"  - {tool}")
    
    print("\n" + "-" * 30)

    try:
        print("Loading YOLOv8 model...")
        YOLOModel = load_model("YOLOv8")
        yolo = YOLOModel()
        print("✓ Model loaded successfully")
    except Exception as e:
        print(f"✗ Failed to load model: {e}")
    
    print("\n" + "-" * 30)
    
    try:
        print("Loading YoloDetect tool...")
        yolo_tool = load_tool("YoloDetect", model_name="YOLOv8", device="cpu")
        print("✓ Tool loaded successfully")
    except Exception as e:
        print(f"✗ Failed to load tool: {e}")

if __name__ == "__main__":
    main()