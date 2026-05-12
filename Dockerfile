# 选用与你本地 WSL2 驱动兼容的 CUDA 运行时基础镜像
FROM nvidia/cuda:13.0.3-cudnn-runtime-ubuntu22.04

WORKDIR /workspace

# 避免 apt 安装时出现时区等交互提示
ENV DEBIAN_FRONTEND=noninteractive

# 安装 Python 和基础系统图形库 (GPU 版的 OpenCV 有时需要这些底层库支持)
RUN apt-get update && apt-get install -y \
    python3-pip \
    python3-dev \
    libgl1-mesa-glx \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# 将系统 python3 链接为 python
RUN ln -s /usr/bin/python3 /usr/bin/python

# 安装带 CUDA 支持相关的包
RUN pip install --upgrade pip && \
    pip install opencv-python numpy torch torchvision
# 安装其他常用的科学计算和绘图库
RUN pip install astropy matplotlib
RUN pip install photutils scipy 
CMD ["tail", "-f", "/dev/null"]