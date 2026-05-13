import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
from scipy.spatial import cKDTree
from astropy.io import fits

def run_cross_match_diagnosis(pipeline_csv, sextractor_csv, fits_image_path, match_radius=2.0):
    print("🔍 启动交叉匹配诊断引擎...")
    
    # 1. 加载数据
    df_mine = pd.read_csv(pipeline_csv)
    # 自动识别空格或逗号分隔
    try:
        df_sex = pd.read_csv(sextractor_csv)
        if len(df_sex.columns) <= 1:
            df_sex = pd.read_csv(sextractor_csv, sep='\s+')
    except:
        df_sex = pd.read_csv(sextractor_csv, sep='\s+')
    
    # 2. 提取坐标矩阵 (🌟 关键修正点)
    # 我方系统列名是 X, Y
    coords_mine = df_mine[['X', 'Y']].values
    
    # SExtractor 示例文件的列名是 X_IMAGE, Y_IMAGE
    if 'X_IMAGE' in df_sex.columns:
        coords_sex = df_sex[['X_IMAGE', 'Y_IMAGE']].values
    elif 'XWIN_IMAGE' in df_sex.columns:
        coords_sex = df_sex[['XWIN_IMAGE', 'YWIN_IMAGE']].values
    else:
        print(f"❌ 错误：在 SE 目录中找不到坐标列。当前列名为: {df_sex.columns.tolist()}")
        return

    print(f" -> 我的系统检出: {len(coords_mine)} 颗")
    print(f" -> SExtractor 检出: {len(coords_sex)} 颗")

    # 3. 构建 KD-Tree 空间索引并匹配
    tree_sex = cKDTree(coords_sex)
    distances, indices = tree_sex.query(coords_mine, k=1)
    
    matched_mask = distances < match_radius
    orphan_mask = ~matched_mask
    
    df_orphans = df_mine[orphan_mask]
    print(f" -> 匹配成功 (双方共识): {np.sum(matched_mask)} 颗")
    print(f" -> 孤儿目标 (我方独有): {len(df_orphans)} 颗")

   # 5. 可视化诊断
    print(f"🎨 正在渲染诊断图像...")
    data = fits.getdata(fits_image_path)
    
    # --- 🌟 关键修复开始 ---
    # 1. 确保数据中没有非正数（对数映射要求值 > 0）
    # 我们将所有小于等于 0 的值替换为一个微小的正数（如背景标准差的 0.1 倍）
    display_data = data.copy()
    floor_val = np.percentile(display_data[display_data > 0], 1) if np.any(display_data > 0) else 1e-5
    display_data[display_data <= 0] = floor_val

    # 2. 安全计算 vmin 和 vmax
    vmin = np.percentile(display_data, 1)
    vmax = np.percentile(display_data, 99.5) # 稍微提高上限，增强对比度
    
    # 3. 防止 vmin 和 vmax 重合
    if vmin >= vmax:
        vmin = display_data.min()
        vmax = display_data.max()
    # --- 🌟 关键修复结束 ---

    plt.figure(figsize=(12, 12))
    
    # 使用修复后的 display_data 和 vmin/vmax
    plt.imshow(display_data, cmap='Greys_r', origin='lower', 
               norm=LogNorm(vmin=vmin, vmax=vmax))
    # 画出那 3000 个异常孤儿点 (红色大圈，重点嫌疑)
    plt.scatter(df_orphans['X'], df_orphans['Y'], 
                s=40, facecolors='none', edgecolors='red', linewidth=1.5, label='Orphans (Pipeline Only)')
    
    plt.title("Cross-Match Diagnostic: Pipeline vs SExtractor", fontsize=16)
    plt.xlabel("X Pixel")
    plt.ylabel("Y Pixel")
    plt.legend(loc='upper right')
    
    # 保存高分辨率诊断图
    out_img = "diagnostic_plot.png"
    plt.savefig(out_img, dpi=300, bbox_inches='tight')
    print(f"✅ 诊断完毕！请立即打开 {out_img} 验尸！")

if __name__ == "__main__":
    # 请填入你实际的文件路径
    PIPELINE_CSV = "output/new/all_raw_detections.csv" # 我们刚才新增导出的 Level 1 总表
    SEXTRACTOR_CSV = "output/sextractor_test/catalogs/20260309133855754_6002_stand.csv" # 你的 SExtractor 跑出来的结果
    FITS_IMAGE = "data/fits_sequence/20260309133855754_6002_stand.fits" # 那张异常的原始 FITS 图片
    
    run_cross_match_diagnosis(PIPELINE_CSV, SEXTRACTOR_CSV, FITS_IMAGE, match_radius=2.0)