import numba
from numba import cuda

print(f"Numba 版本: {numba.__version__}")
try:
    # 尝试检测 CUDA 设备
    gpu_detected = cuda.is_available()
    print(f"Numba.cuda 是否识别到 GPU: {gpu_detected}")
except Exception as e:
    print(f"CUDA 检测异常: {e}")