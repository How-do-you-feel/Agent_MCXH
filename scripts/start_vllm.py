import subprocess
import sys
import os
import time

def start_vllm_server(model_path="/home/ps/Qwen2.5-3B", host="0.0.0.0", port=8000, 
                      gpu_memory_utilization=0.8, max_model_len=None):
    """Start vLLM server with specified model"""
    
    if not os.path.exists(model_path):
        print(f"Error: Model path {model_path} does not exist")
        return False
    
    cmd = [
        "python", "-m", "vllm.entrypoints.openai.api_server",
        "--model", model_path,
        "--host", host,
        "--port", str(port),
        "--gpu-memory-utilization", str(gpu_memory_utilization)
    ]
    
    # 如果设置了最大模型长度，则添加该参数
    if max_model_len is not None:
        cmd.extend(["--max-model-len", str(max_model_len)])
    
    print(f"Starting vLLM server with model: {model_path}")
    print(f"Host: {host}, Port: {port}")
    print(f"GPU memory utilization: {gpu_memory_utilization}")
    if max_model_len is not None:
        print(f"Max model length: {max_model_len}")
    
    try:
        process = subprocess.Popen(cmd)
        print(f"vLLM server started with PID: {process.pid}")
        
        # Wait a moment for server to start
        time.sleep(2)
        
        # Check if process is still running
        if process.poll() is None:
            print("vLLM server is running")
            return True
        else:
            print("vLLM server failed to start")
            return False
            
    except Exception as e:
        print(f"Failed to start vLLM server: {e}")
        return False

def main():
    import argparse
    
    parser = argparse.ArgumentParser(description='Start vLLM server')
    parser.add_argument('--model', type=str, default='/home/ps/Qwen2.5-3B', help='Model path')
    parser.add_argument('--host', type=str, default='0.0.0.0', help='Host to bind to')
    parser.add_argument('--port', type=int, default=8000, help='Port to bind to')
    parser.add_argument('--gpu-memory-utilization', type=float, default=0.8, 
                        help='GPU memory utilization (0.0-1.0)')
    parser.add_argument('--max-model-len', type=int, default=None, 
                        help='Maximum model length')
    
    args = parser.parse_args()
    
    success = start_vllm_server(args.model, args.host, args.port, 
                                args.gpu_memory_utilization, args.max_model_len)
    if not success:
        sys.exit(1)

if __name__ == "__main__":
    main()