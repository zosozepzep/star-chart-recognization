import os
import warnings
import numpy as np
import matplotlib
# 设置无头(headless)渲染后端，适配 Docker 等服务器环境
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from astropy.io import fits
from astropy.utils.exceptions import AstropyWarning
from matplotlib.colors import LogNorm

# 屏蔽非致命的 Astropy 解析警告
warnings.simplefilter('ignore', category=AstropyWarning)


def inspect_potential_star(fits_path, target_x, target_y, output_dir, patch_size=15):
    """
    无头环境适配版的时序片段 3D 渲染与物理特性检查
    """
    print(f"🔍 启动物理显微分析: 锁定坐标 [{target_x:.2f}, {target_y:.2f}]")
    
    with fits.open(fits_path, ignore_missing_end=True) as hdul:
        data = hdul[0].data.astype(float)
        
    cx, cy = int(round(target_x)), int(round(target_y))
    y_slice = slice(max(0, cy - patch_size//2), min(data.shape[0], cy + patch_size//2 + 1))
    x_slice = slice(max(0, cx - patch_size//2), min(data.shape[1], cx + patch_size//2 + 1))
    
    patch_data = data[y_slice, x_slice]
    
    if patch_data.size == 0:
        print("❌ 错误: 目标超出图片物理边界。")
        return

    # === 高清渲染设置 ===
    # 调大分辨率，方便在 VS Code 里放大看细节
    fig = plt.figure(figsize=(16, 7), dpi=150) 
    plt.suptitle(f"Physical Target Inspection [X: {target_x:.1f}, Y: {target_y:.1f}]", fontsize=16)

    # 1. 2D 对数拉伸显微照片
    ax1 = fig.add_subplot(121)
    ax1.set_title("16-bit Raw Data Slice (Log Scale)")
    ax1.scatter(patch_size//2, patch_size//2, color='red', marker='+', s=200, linewidth=2)
    im1 = ax1.imshow(patch_data, origin='lower', cmap='viridis', 
                    norm=LogNorm(vmin=np.median(data), vmax=np.max(patch_data)))
    fig.colorbar(im1, ax=ax1, fraction=0.046, pad=0.04)

    # 2. 3D 能量峰分布图
    ax2 = fig.add_subplot(122, projection='3d')
    ax2.set_title("3D Energy Flux Topography")
    
    # 优化网格生成 (使用 mgrid 直接创建)
    height, width = patch_data.shape
    X, Y = np.mgrid[0:width, 0:height]
    
    # 渲染 3D 曲面
    surf = ax2.plot_surface(X, Y, patch_data.T, cmap='inferno', edgecolor='none', alpha=0.9)
    ax2.set_zlabel('Photon Flux (ADU)', fontsize=10, labelpad=10)
    # 调整观看视角，立体感更强
    ax2.view_init(elev=30, azim=45) 

    # 将图片写入磁盘
    images_dir = os.path.join(output_dir, "images")
    os.makedirs(images_dir, exist_ok=True)
    out_file = os.path.join(images_dir, f"3D_Inspect_X{int(target_x)}_Y{int(target_y)}.jpg")
    
    # 解决边缘可能被裁切的问题
    plt.tight_layout() 
    plt.savefig(out_file, bbox_inches='tight')
    plt.close(fig)  # 释放内存
    
    print(f"✅ 渲染完成！请在相对目录打开查看: {out_file}")


if __name__ == "__main__":
    # 单元测试配置路径
    WORKSPACE = "/workspace"
    FITS_FILE = os.path.join(WORKSPACE, "data/images/img_01.fits")
    OUT_DIR = os.path.join(WORKSPACE, "output")
    
    # 待校验的疑似噪点或目标物理坐标
    test_x = 2994.62
    test_y = 1072.98
    
    inspect_potential_star(FITS_FILE, test_x, test_y, OUT_DIR)