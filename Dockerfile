# 智能客服 RAG 系统 Dockerfile

FROM python:3.11-slim

WORKDIR /app

# 安装系统依赖（PyMuPDF 需要）
RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc \
    g++ \
    libffi-dev \
    && rm -rf /var/lib/apt/lists/*

# 安装 Python 依赖
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt -i https://pypi.tuna.tsinghua.edu.cn/simple

# 复制应用代码
COPY src/ ./src/

# 创建数据目录
RUN mkdir -p /app/data/documents

# 暴露端口
EXPOSE 8000

# 启动服务
CMD ["python", "-m", "src.main"]