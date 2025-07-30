# /home/ps/MCXH/Agent_MCXH/examples/simple_vision_chat.py
#!/usr/bin/env python3
"""简单的视觉智能体终端交互示例"""

import sys
import os

# 添加项目根目录到Python路径
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from prompt_toolkit import ANSI, prompt
from ag_mcxh import VisionAgent

def main():
    print("视觉智能体终端交互示例")
    print("=" * 40)
    
    # 初始化视觉智能体
    try:
        agent = VisionAgent(
            model_path="/home/ps/Qwen2.5-3B",
            host="127.0.0.1",
            port=8001
        )
        print("✓ 视觉智能体初始化成功")
    except Exception as e:
        print(f"✗ 视觉智能体初始化失败: {e}")
        return

    # 默认图像路径
    default_image_path = "/home/ps/MCXH/MingChaXinHao/ag_mcxh/pics/002.png"
    current_image_path = default_image_path
    
    print(f"\n默认图像路径: {current_image_path}")
    print("命令:")
    print("  'set_image <路径>' - 设置新的图像路径")
    print("  'detect' - 使用YOLO检测图像中的对象")
    print("  'segment' - 使用SegmentAnything分割图像")
    print("  'ask <问题>' - 向智能体提问关于图像的问题")
    print("  'exit' - 退出程序")
    print("-" * 40)

    while True:
        try:
            user_input = prompt(ANSI('\033[92mUser\033[0m: ')).strip()
        except UnicodeDecodeError:
            print('输入错误')
            continue
        except KeyboardInterrupt:
            print("\n再见!")
            break
            
        if not user_input:
            continue
            
        if user_input.lower() == 'exit':
            print("再见!")
            break
            
        # 处理命令
        if user_input.startswith('set_image '):
            new_path = user_input[10:].strip()
            if os.path.exists(new_path):
                current_image_path = new_path
                print(f"图像路径已更新为: {current_image_path}")
            else:
                print(f"图像文件不存在: {new_path}")
                
        elif user_input == 'detect':
            if not os.path.exists(current_image_path):
                print(f"图像文件不存在: {current_image_path}")
                continue
                
            try:
                print("正在使用YOLO检测图像...")
                result = agent.direct_tool_call("YoloDetect", current_image_path, device="cpu")
                print(f'\033[91mYOLO检测结果\033[0m: {result}')
            except Exception as e:
                print(f'\033[91m检测失败\033[0m: {e}')
                
        elif user_input == 'segment':
            if not os.path.exists(current_image_path):
                print(f"图像文件不存在: {current_image_path}")
                continue
                
            try:
                print("正在使用SegmentAnything分割图像...")
                result = agent.direct_tool_call("SegmentAnything", current_image_path, device="cpu")
                print(f'\033[91m分割结果\033[0m: {result}')
            except Exception as e:
                print(f'\033[91m分割失败\033[0m: {e}')
                
        elif user_input.startswith('ask '):
            if not os.path.exists(current_image_path):
                print(f"图像文件不存在: {current_image_path}")
                continue
                
            question = user_input[4:].strip()
            if not question:
                print("请输入问题")
                continue
                
            try:
                print("正在处理问题...")
                result = agent.process_with_vllm(question, current_image_path)
                print(f'\033[91m智能体回答\033[0m: {result}')
            except Exception as e:
                print(f'\033[91m处理失败\033[0m: {e}')
                
        else:
            print("未知命令。可用命令: set_image, detect, segment, ask, exit")

if __name__ == "__main__":
    main()