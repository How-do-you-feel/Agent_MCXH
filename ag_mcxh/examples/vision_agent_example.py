import sys
import os
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

def main():
    try:
        from ag_mcxh import VisionAgent
        
        agent = VisionAgent()
        
        image_path = "/home/ps/MCXH/MingChaXinHao/ag_mcxh/pics/002.png"
        prompt = "Detect all objects in the image"
        
        if not os.path.exists(image_path):
            print("Image file does not exist")
            return
            
        result = agent.process(prompt, image_path)
        print(result)
        
    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    main()