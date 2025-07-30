from typing import List, Dict, Any, Optional
import requests
import json
import base64
import time

class VLLMHTTPClient:
    """通过HTTP API与vLLM服务器通信的客户端"""
    
    def __init__(self, base_url: str = "http://localhost:8000"):
        self.base_url = base_url.rstrip('/')
        
    def encode_image(self, image_path: str) -> str:
        """将图像编码为base64字符串"""
        with open(image_path, "rb") as image_file:
            return base64.b64encode(image_file.read()).decode('utf-8')
    
    def generate(self, prompt: str, **kwargs) -> str:
        """通过HTTP API生成文本响应"""
        # 首先尝试使用OpenAI兼容的API端点
        url = f"{self.base_url}/v1/completions"
        
        # 默认参数
        data = {
            "model": kwargs.get("model", ""),  # 模型名称
            "prompt": prompt,
            "temperature": kwargs.get("temperature", 0.7),
            "top_p": kwargs.get("top_p", 0.9),
            "max_tokens": kwargs.get("max_tokens", 512),
            "stream": False
        }
        
        # 添加其他参数
        for key, value in kwargs.items():
            if key not in data:
                data[key] = value
        
        try:
            response = requests.post(
                url, 
                json=data,
                headers={"Content-Type": "application/json"}
            )
            response.raise_for_status()
            result = response.json()
            return result["choices"][0]["text"].strip()
        except Exception as e:
            # 如果v1/completions不可用，尝试使用原始的generate端点
            try:
                legacy_url = f"{self.base_url}/generate"
                legacy_data = {
                    "prompt": prompt,
                    "temperature": kwargs.get("temperature", 0.7),
                    "top_p": kwargs.get("top_p", 0.9),
                    "max_tokens": kwargs.get("max_tokens", 512),
                }
                response = requests.post(legacy_url, json=legacy_data)
                response.raise_for_status()
                result = response.json()
                return result["text"][0].strip()
            except Exception as legacy_e:
                raise RuntimeError(f"Failed to generate text with both new and legacy APIs: {str(e)}, {str(legacy_e)}")
    
    def chat_completion(self, messages: List[Dict[str, Any]], **kwargs) -> str:
        """使用OpenAI兼容的聊天完成接口"""
        url = f"{self.base_url}/v1/chat/completions"
        
        # 构造请求数据
        data = {
            "model": kwargs.get("model", ""),
            "messages": messages,
            "temperature": kwargs.get("temperature", 0.7),
            "top_p": kwargs.get("top_p", 0.9),
            "max_tokens": kwargs.get("max_tokens", 512),
            "stream": False
        }
        
        try:
            response = requests.post(
                url,
                json=data,
                headers={"Content-Type": "application/json"}
            )
            response.raise_for_status()
            result = response.json()
            return result["choices"][0]["message"]["content"].strip()
        except Exception as e:
            # 回退到模拟实现
            prompt_parts = []
            for msg in messages:
                if msg["role"] == "system":
                    prompt_parts.append(f"System: {msg['content']}")
                elif msg["role"] == "user":
                    prompt_parts.append(f"User: {msg['content']}")
                elif msg["role"] == "assistant":
                    prompt_parts.append(f"Assistant: {msg['content']}")
            
            prompt = "\n".join(prompt_parts) + "\nAssistant:"
            return self.generate(prompt, **kwargs)

class VLLMAsyncClient:
    """异步启动和管理vLLM服务器的客户端"""
    
    def __init__(self, model_path: str, host: str = "127.0.0.1", port: int = 8001, 
                 gpu_memory_utilization: float = 0.8, max_model_len: Optional[int] = None):
        self.model_path = model_path
        self.host = host
        self.port = port
        self.gpu_memory_utilization = gpu_memory_utilization
        self.max_model_len = max_model_len
        self.base_url = f"http://{host}:{port}"
        self.process = None
        
    def start_server(self):
        """启动vLLM服务器"""
        try:
            import subprocess
            import os
            
            # 尝试使用OpenAI兼容的API服务器入口点
            cmd = [
                "python", "-m", "vllm.entrypoints.openai.api_server",
                "--host", self.host,
                "--port", str(self.port),
                "--model", self.model_path,
                "--gpu-memory-utilization", str(self.gpu_memory_utilization)
            ]
            
            # 如果设置了最大模型长度，则添加该参数
            if self.max_model_len is not None:
                cmd.extend(["--max-model-len", str(self.max_model_len)])
            
            # 启动服务器进程
            self.process = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE
            )
            
            # 等待服务器启动
            self._wait_for_server()
            
            print(f"vLLM server started on {self.base_url}")
            return self.process.pid
            
        except Exception as e:
            # 如果OpenAI兼容的入口点失败，回退到原始入口点
            try:
                print(f"Failed to start with OpenAI API server, falling back to legacy API server: {e}")
                cmd = [
                    "python", "-m", "vllm.entrypoints.api_server",
                    "--host", self.host,
                    "--port", str(self.port),
                    "--model", self.model_path,
                    "--gpu-memory-utilization", str(self.gpu_memory_utilization)
                ]
                
                # 如果设置了最大模型长度，则添加该参数
                if self.max_model_len is not None:
                    cmd.extend(["--max-model-len", str(self.max_model_len)])
                
                # 启动服务器进程
                self.process = subprocess.Popen(
                    cmd,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE
                )
                
                # 等待服务器启动
                self._wait_for_server()
                
                print(f"vLLM server started on {self.base_url}")
                return self.process.pid
            except Exception as fallback_e:
                raise RuntimeError(f"Failed to start vLLM server with both entrypoints: {str(e)}, {str(fallback_e)}")
    
    def _wait_for_server(self, timeout: int = 30):
        """等待服务器启动"""
        start_time = time.time()
        while time.time() - start_time < timeout:
            try:
                response = requests.get(f"{self.base_url}/health")
                if response.status_code == 200:
                    return
            except:
                pass
            time.sleep(1)
        raise RuntimeError("vLLM server failed to start within timeout")
    
    def stop_server(self):
        """停止vLLM服务器"""
        if self.process and self.process.poll() is None:
            self.process.terminate()
            self.process.wait()
            print("vLLM server stopped")
    
    def get_client(self) -> VLLMHTTPClient:
        """获取HTTP客户端"""
        return VLLMHTTPClient(self.base_url)