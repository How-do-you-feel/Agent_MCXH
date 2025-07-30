#!/bin/bash

# 启动开发环境

# 检查是否安装了必要的依赖
if ! command -v npm >/dev/null 2>&1
then
    echo "npm 未安装，请先安装 Node.js 和 npm"
    exit 1
fi

echo "npm 已安装，路径: $(which npm)"

# 检查是否安装了Python依赖
if ! python3 -c "import flask" >/dev/null 2>&1
then
    echo "安装Python依赖..."
    pip install -r requirements.txt
fi

# 在后台启动后端
echo "启动后端服务..."
python3 app.py &
BACKEND_PID=$!

# 等待后端服务启动
echo "等待后端服务启动..."
sleep 8

# 检查后端是否成功启动
if ps -p $BACKEND_PID >/dev/null 2>&1
then
    echo "后端服务启动成功 (PID: $BACKEND_PID)"
else
    echo "后端服务启动失败"
    exit 1
fi

# 测试后端API是否可访问
echo "测试后端API..."
curl -s http://localhost:5000/api/tools > /dev/null
if [ $? -eq 0 ]; then
    echo "后端API测试成功"
else
    echo "后端API测试失败，但继续启动前端"
fi

# 在前台启动前端开发服务器
echo "启动前端开发服务器..."
cd frontend
npm run dev

# 当前端服务结束时，终止后端服务
echo "正在终止后端服务..."
kill $BACKEND_PID 2>/dev/null