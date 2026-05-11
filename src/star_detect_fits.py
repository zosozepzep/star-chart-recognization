import cv2
import numpy as np
import json
import os
from astropy.io import fits

def read_fits_to_cv2(fits_path):
    """
    读取天文 FITS 文件，并将其归一化为 OpenCV 可处理的 8 位灰度图
    """
    try:
        # 读取图像数据矩阵和头部元数据
        data, header = fits.getdata(fits_path, header=True)
    except Exception as e:
        print(f"❌ FITS 读取失败，请检查路径或文件损坏: {e}")
        return None, None

    # 异常值剔除：排除 1% 最暗噪点和 99.9% 极亮噪点（如宇宙射线）
    p_min, p_max = np.percentile(data, (1, 99.9))
    
    # 截断异常值并线性映射到 0-1 之间
    data_clipped = np.clip(data, p_min, p_max)
    normalized_data = (data_clipped - p_min) / (p_max - p_min)
    
    # 转换为 0-255 的标准 8 位无符号整数
    img_8u = (normalized_data * 255.0).astype(np.uint8)

    return img_8u, header

def process_star_map(fits_path, output_dir, file_name):
    # 1. 加载 FITS 并转换为灰度矩阵
    img, header = read_fits_to_cv2(fits_path)
    if img is None:
        return

    # 复制一份彩色图，用于最终给裁判看的视觉呈现
    display_img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)

    # 2. 预处理：高斯滤波去底噪
    blurred = cv2.GaussianBlur(img, (3, 3), 0)

  # 3. 二值化分割：动态统计阈值法 (Dynamic Thresholding)
    # 计算当前模糊图像的全局平均灰度 (mean) 和 标准差 (std)
    mean_val, std_val = cv2.meanStdDev(blurred)
    mean_val = mean_val[0][0]
    std_val = std_val[0][0]

    # 设定阈值为：背景平均亮度 + 5倍的亮度波动(标准差)
    # 这里的 5 是一个更具鲁棒性的超参数。如果噪点还是多，改成 6 或 7；如果丢星了，改成 3 或 4。
    dynamic_threshold = mean_val + 5 * std_val
    print(f"📊 本图背景均值: {mean_val:.1f}, 标准差: {std_val:.1f}")
    print(f"🎯 自动计算阈值设定为: {dynamic_threshold:.1f}")

    _, thresh = cv2.threshold(blurred, dynamic_threshold, 255, cv2.THRESH_BINARY)

    # 4. 连通域分析：圈出独立星点
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    stars_data = []
    dimmest_star = None
    min_intensity = float('inf')

    # 5. 遍历计算物理属性
    for i, contour in enumerate(contours):
        area = cv2.contourArea(contour)
# 过滤掉极小的噪点，以及极其巨大的背景块或边缘伪影
        if area < 1.5 or area > 1000:
            continue

        # 计算几何中心 (质心)
        M = cv2.moments(contour)
        if M["m00"] != 0:
            cx = int(M["m10"] / M["m00"])
            cy = int(M["m01"] / M["m00"])
        else:
            cx, cy = contour[0][0][0], contour[0][0][1]

        # 计算区域内的能量积分 (代表星等代理值)
        mask = np.zeros_like(img)
        cv2.drawContours(mask, [contour], -1, 255, -1)
        mean_val = cv2.mean(img, mask=mask)[0]
        intensity = mean_val * area

        star_info = {
            "id": i + 1,
            "x": cx,
            "y": cy,
            "pixel_area": round(area, 1),
            "intensity": round(intensity, 2)
        }
        stars_data.append(star_info)

        # 寻找最暗目标
        if intensity < min_intensity:
            min_intensity = intensity
            dimmest_star = star_info

        # 【可视化】画绿圈
        cv2.drawContours(display_img, [contour], -1, (0, 255, 0), 1)

    # 6. 【可视化】标注最暗的星
    if dimmest_star:
        x, y = dimmest_star["x"], dimmest_star["y"]
        cv2.rectangle(display_img, (x-10, y-10), (x+10, y+10), (0, 0, 255), 2)
        cv2.putText(display_img, f"Dimmest ID:{dimmest_star['id']}", 
                    (x + 15, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)

    total_stars = len(stars_data)
    cv2.putText(display_img, f"Total Stars Found: {total_stars}", (20, 40), 
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    # 7. 提取创新加分项：FITS 头部元数据
    # 尝试读取曝光时间、观测日期等关键物理信息，若无则为空
    extracted_meta = {
        "exposure_time": header.get("EXPTIME", "N/A"),
        "observation_date": header.get("DATE-OBS", "N/A"),
        "instrument": header.get("INSTRUME", "N/A")
    }

    # 8. 持久化输出
    os.makedirs(output_dir, exist_ok=True)
    base_name = os.path.splitext(file_name)[0]
    
    # 写入图像
    out_img_path = os.path.join(output_dir, f"result_{base_name}.jpg")
    cv2.imwrite(out_img_path, display_img)

    # 写入 JSON
    out_json_path = os.path.join(output_dir, f"data_{base_name}.json")
    output_payload = {
        "task": "star_recognition_baseline",
        "source_file": file_name,
        "metadata": extracted_meta,
        "summary": {
            "total_stars_count": total_stars,
            "dimmest_star_target": dimmest_star
        },
        "all_stars_list": stars_data
    }
    
    with open(out_json_path, 'w', encoding='utf-8') as f:
        json.dump(output_payload, f, indent=4)

    print(f"✅ 处理完毕: {file_name}")
    print(f"   -> 发现星点: {total_stars}")
    print(f"   -> 结果图: {out_img_path}")
    print(f"   -> 数据集: {out_json_path}")

if __name__ == "__main__":
    # 配置目录映射路径
    WORK_DIR = "/workspace"
    IMAGE_DIR = os.path.join(WORK_DIR, "data/images")
    OUT_DIR = os.path.join(WORK_DIR, "output")
    
    # 假设你的比赛数据命名为 img_01.fits
    target_file = "img_01.fits" 
    file_path = os.path.join(IMAGE_DIR, target_file)
    
    if not os.path.exists(file_path):
        print(f"⚠️ 找不到测试文件: {file_path}")
        print("请确保已将大赛的 .fits 文件放入 data/images/ 目录下，并修改 target_file 变量名。")
    else:
        process_star_map(file_path, OUT_DIR, target_file)