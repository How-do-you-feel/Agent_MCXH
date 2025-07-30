from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
import os
import sys
import tempfile
from werkzeug.utils import secure_filename

# 添加项目根目录到Python路径
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ag_mcxh_path = os.path.join(project_root, 'ag_mcxh')

# 确保ag_mcxh路径在sys.path中
if ag_mcxh_path not in sys.path:
    sys.path.insert(0, ag_mcxh_path)
if project_root not in sys.path:
    sys.path.insert(0, project_root)

# 导入VisionAgent
try:
    from ag_mcxh import VisionAgent
    print("成功导入VisionAgent")
except ImportError as e:
    print(f"导入VisionAgent失败: {e}")
    VisionAgent = None

# 获取当前文件所在目录
current_dir = os.path.dirname(os.path.abspath(__file__))
# 设置静态文件目录
static_folder = os.path.join(current_dir, 'dist')

app = Flask(__name__, static_folder=static_folder)
CORS(app)

# 配置
UPLOAD_FOLDER = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'uploads')
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'gif'}
MAX_CONTENT_LENGTH = 16 * 1024 * 1024  # 16MB max file size

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['MAX_CONTENT_LENGTH'] = MAX_CONTENT_LENGTH

# 确保上传目录存在
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# 全局变量存储VisionAgent实例
agent = None

def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route('/api/init', methods=['POST'])
def init_vision_agent():
    global agent
    
    # 如果已经有agent实例，先停止它
    if agent is not None:
        try:
            del agent
        except:
            pass
    
    data = request.json
    if not data:
        return jsonify({'error': '缺少请求数据'}), 400
        
    model_path = data.get('model_path')
    host = data.get('host')
    port = data.get('port')

    if not model_path or not host or not port:
        return jsonify({'error': '缺少必要参数'}), 400

    try:
        # 初始化VisionAgent
        agent = VisionAgent(model_path=model_path, host=host, port=port)
        return jsonify({'message': '初始化成功'})
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/process', methods=['POST'])
def process_image():
    global agent
    if agent is None:
        return jsonify({'error': 'VisionAgent 未初始化'}), 500
    
    # 检查是否有文件部分
    if 'image' not in request.files:
        return jsonify({'error': '没有图像文件'}), 400
    
    file = request.files['image']
    
    # 检查是否有文件被选择
    if file.filename == '':
        return jsonify({'error': '没有选择文件'}), 400
    
    # 检查文件类型
    if not allowed_file(file.filename):
        return jsonify({'error': '不支持的文件类型'}), 400
    
    # 获取提示和工具名称
    prompt = request.form.get('prompt', '')
    tool_name = request.form.get('tool_name', '')
    
    if not prompt:
        return jsonify({'error': '提示不能为空'}), 400
    
    try:
        # 保存文件到临时位置
        filename = secure_filename(file.filename)
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)
        
        # 处理图像
        if tool_name:
            # 直接调用指定工具
            result = agent.direct_tool_call(tool_name, filepath, device="cpu")
        else:
            # 使用vLLM自动选择工具
            result = agent.process_with_vllm(prompt, filepath)
        
        # 清理上传的文件
        os.remove(filepath)
        
        return jsonify({'result': result})
    
    except Exception as e:
        # 确保即使出错也清理文件
        if 'filepath' in locals() and os.path.exists(filepath):
            os.remove(filepath)
        return jsonify({'error': str(e)}), 500

@app.route('/api/tools', methods=['GET'])
def get_tools():
    """获取可用工具列表"""
    global agent
    if agent is None:
        return jsonify({'error': 'VisionAgent 未初始化'}), 500
    
    try:
        tools = agent.tool_selector.get_available_tools()
        tool_list = []
        for tool_name in tools:
            # 这里可以扩展以包含工具的详细描述
            tool_list.append({
                'name': tool_name,
                'description': f'{tool_name} 工具'
            })
        return jsonify({'tools': tool_list})
    except Exception as e:
        return jsonify({'error': str(e)}), 500

# Serve frontend static files
@app.route('/', defaults={'path': ''})
@app.route('/<path:path>')
def serve_frontend(path):
    # 如果请求的是API，则不处理
    if path.startswith('api/'):
        return jsonify({'error': 'API endpoint not found'}), 404
        
    # 如果dist目录存在
    if app.static_folder and os.path.exists(app.static_folder):
        # 如果请求的文件存在，则返回该文件
        if path != "" and os.path.exists(os.path.join(app.static_folder, path)):
            return send_from_directory(app.static_folder, path)
        else:
            # 否则返回index.html（适用于单页应用）
            return send_from_directory(app.static_folder, 'index.html')
    else:
        # 如果没有构建好的前端文件，则返回简单的消息
        if path == "":
            return "Vision Agent WebUI Backend is running. Please build the frontend or use development mode."
        else:
            return jsonify({'error': 'File not found'}), 404

@app.route('/health')
def health_check():
    """健康检查端点"""
    return jsonify({
        'status': 'ok',
        'vision_agent': agent is not None,
        'message': 'Vision Agent WebUI Backend is running'
    })

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)