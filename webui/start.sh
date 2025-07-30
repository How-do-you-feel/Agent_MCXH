if ! command -v npm &> /dev/null
then
    echo "npm 未安装，请先安装 Node.js 和 npm"
    exit 1
fi

# 检查是否安装了Python依赖
if ! python -c "import flask" &> /dev/null
then
    echo "安装Python依赖..."
    pip install -r requirements.txt
fi

# 构建前端
echo "构建前端..."
cd frontend
npm install
npm run build

# 返回上一级目录
cd ..

# 启动后端服务
echo "启动后端服务..."
python app.py