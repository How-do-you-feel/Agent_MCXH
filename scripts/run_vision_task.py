import sys
import os
import argparse

# Add project root to Python path
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

def main():
    parser = argparse.ArgumentParser(description='Run vision task with natural language prompt')
    parser.add_argument('--prompt', type=str, required=True, help='Natural language prompt')
    parser.add_argument('--image', type=str, required=True, help='Path to image file')
    parser.add_argument('--model', type=str, default='/home/ps/Qwen2.5-3B', help='Model path')
    parser.add_argument('--api-base', type=str, default='http://localhost:8000/v1', help='API base URL')
    
    args = parser.parse_args()
    
    # Validate image path
    if not os.path.exists(args.image):
        print(f"Error: Image file {args.image} does not exist")
        return 1
    
    try:
        from ag_mcxh import VisionAgent
        
        # Create vision agent
        agent = VisionAgent(api_base=args.api_base, model_name=args.model)
        
        # Process the task
        result = agent.process(args.prompt, args.image)
        
        # Output result
        print(result)
        return 0
        
    except Exception as e:
        print(f"Error: {e}")
        return 1

if __name__ == "__main__":
    sys.exit(main())