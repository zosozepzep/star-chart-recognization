import os
import json
import warnings
import cv2
import numpy as np
from scipy.spatial import cKDTree
from astropy.io import fits
from astropy.utils.exceptions import AstropyWarning
from astropy.stats import sigma_clipped_stats
from photutils.background import Background2D, MedianBackground
from photutils.detection import DAOStarFinder, find_peaks
from photutils.segmentation import detect_sources, SourceCatalog
from scipy.stats import linregress

# 屏蔽 Astropy 非致命警告
warnings.simplefilter('ignore', category=AstropyWarning)


def auto_calibrate_and_get_gradients(data_sub, std):
    """
    基于局部加速度监控与物理护栏的梯度校准
    """
    print("🚀 启动动态下界探测引擎...")

    lower_bound = 3.0 # 理论兜底值
    prev_count = 1    # 避免除以 0
    prev_growth_rate = 1.0
    
    # 强制护栏：在此阈值之上，无论数据如何跳变，绝不停机
    SAFETY_GUARDRAIL = 3.5 

    # 从 6.0 往下探，步长 0.15 保证平滑
    for s in np.arange(6.0, 1.9, -0.1):
        tbl = find_peaks(data_sub, s * std, box_size=5)
        current_count = len(tbl) if tbl is not None else 1
        
        # 1. 计算当前步的一阶增长率
        current_growth_rate = current_count / prev_count
        
        # 2. 计算二阶加速度 (当前增长率 / 上一步增长率)
        acceleration = current_growth_rate / prev_growth_rate
        
        # 3. 核心审判逻辑
        if s <= SAFETY_GUARDRAIL:
            # 只有进入深水区 (<= 3.5σ)，才允许触发刹车
            # 如果加速度突然飙升 (超过 2.0倍)，或者单步增长率极度异常 (超过 4.0倍)
            if acceleration > 2.0 or current_growth_rate > 4.0:
                lower_bound = s + 0.1 # 退回安全区
                print(f"   [紧急制动] 突破护栏后遭遇底噪墙！")
                print(f"   -> 阈值: {s:.2f}σ | 恒星数: {current_count} | 加速度: {acceleration:.2f}x")
                print(f"   -> 最终锁定安全下界: {lower_bound:.2f}σ")
                break

        # 状态流转
        prev_count = current_count
        prev_growth_rate = current_growth_rate if current_growth_rate > 0 else 1.0

    # 如果循环自然结束都没有触发报警，说明图片极度干净
    if lower_bound == 3.0 and s < 2.5:
        lower_bound = s + 0.1
        print(f"   [探底成功] 未遭遇严重噪声墙，锁定极限下界: {lower_bound:.2f}σ")

    # ------------------------------------------------
    # 构造加密步进梯度
    # ------------------------------------------------
    upper_bound = 15.0
    gradients = np.unique(np.concatenate([
        np.linspace(upper_bound, 6.0, 10),       # 高亮区
        np.arange(6.0, lower_bound - 0.05, -0.1) # 暗弱区高密度采样
    ]))
    gradients = sorted(gradients, reverse=True)
    
    return gradients, lower_bound


def process_star_map_ultimate(fits_path, img_out_dir, json_out_dir, file_name):
    print(f"\n [高级梯度版] 启动分析流水线: {file_name}")
    
    # ==========================================
    # 模块 1：数据加载与 2D 背景平坦化
    # ==========================================
    try:
        data, header = fits.getdata(fits_path, header=True)
        data = data.astype(float)
        H, W = data.shape 
    except Exception as e:
        print(f"❌ 读取失败: {e}")
        return

    print("⏳ 1/5 构建大网格 2D 背景底噪图 (防止局部亮斑误伤)...")
    bkg_estimator = MedianBackground()
    # 使用 256x256 的大视野尺寸保护微弱的中等星源不被当作底层平场除掉
    bkg = Background2D(data, (256, 256), filter_size=(3, 3), bkg_estimator=bkg_estimator)
    data_sub = data - bkg.background
    _, _, std = sigma_clipped_stats(data_sub, sigma=3.0)

    # ==========================================
    # 模块 2：锚定测度与 FWHM 预估
    # ==========================================
    # 同时接收梯度列表和数学下界
    dynamic_gradients, lower_bound = auto_calibrate_and_get_gradients(data_sub, std)
    
    safe_high_threshold = dynamic_gradients[0] * std
    segm_calib = detect_sources(data_sub, safe_high_threshold, npixels=5)
    
    dynamic_fwhm = 3.0
    if segm_calib is not None:
        cat_calib = SourceCatalog(data_sub, segm_calib)
        fwhm_data = getattr(cat_calib.fwhm, 'value', cat_calib.fwhm)
        valid_fwhm = fwhm_data[~np.isnan(fwhm_data)]
        if len(valid_fwhm) > 0:
            dynamic_fwhm = float(np.median(valid_fwhm))
    print(f"   -> 锚点测量完毕，当前星空 FWHM: {dynamic_fwhm:.2f} px")

    # ==========================================
    # 模块 3：分布下降扫描引擎
    # ==========================================
    print("⏳ 3/5 启动多级梯度剥离探测引擎...")
    global_stars = []
    global_coords = []

    for i, grad in enumerate(dynamic_gradients):
        current_threshold = grad * std
        
        # 针对当前安全维度设定过滤体积
        if grad > 10.0: 
            npixels = 8
        elif grad > 6.0: 
            npixels = 6
        else: 
            npixels = 4
            
        print(f"   -> [Layer {i+1}] 扫测阈值 {grad:.2f}σ | 最小体积 {npixels}px")
        
        tier_candidates = []
        
        # ==========================================
        # 保护机制微观引擎 B：提取解析点散布模型的细节参量
        # ==========================================
        daofind = DAOStarFinder(
            fwhm=dynamic_fwhm, 
            threshold=current_threshold,
            sharplo=0.3,  
            sharphi=0.95  
        )
        sources_psf = daofind(data_sub)
        if sources_psf is not None:
            for row in sources_psf:
                # 获取星源解析特有的 PSF 内在光度
                flux = float(row['flux'])
                mag = float(row['mag']) if not np.isnan(row['mag']) else 0.0
                sharpness = float(row['sharpness'])
                roundness = float(row['roundness1'])  # roundness=0 为理论正圆
                
                tier_candidates.append({
                    "centroid": {
                        "x": round(float(row['xcentroid']), 3), 
                        "y": round(float(row['ycentroid']), 3)
                    },
                    "photometry": {
                        "peak": round(float(row['peak']), 3), 
                        "flux": round(flux, 3), 
                        "mag": round(mag, 3)
                    },
                    "morphology": {
                        "sharpness": round(sharpness, 3), 
                        "roundness": round(roundness, 3),
                        "elongation": 1.0,  # PSF 标准模型视作固定非延展对象 1.0
                        "orientation": 0.0,
                        "equivalent_radius": round(dynamic_fwhm / 2.0, 3)
                    },
                    "meta": {"source": "Engine_B_Micro", "tier": i + 1}
                })

        # ==========================================
        # 保护机制引擎 A：获取源特征的宏观边界与光照形态
        # ==========================================
        segm = detect_sources(data_sub, current_threshold, npixels=npixels)
        if segm is not None:
            cat = SourceCatalog(data_sub, segm)
            for obj in cat:
                ecc = float(getattr(obj.eccentricity, 'value', obj.eccentricity))
                
                if ecc < 0.85:
                    peak = float(getattr(obj.max_value, 'value', obj.max_value))
                    flux = float(getattr(obj.segment_flux, 'value', obj.segment_flux))
                    elong = float(getattr(obj.elongation, 'value', obj.elongation))
                    theta = float(getattr(obj.orientation, 'value', obj.orientation))
                    eq_radius = float(getattr(obj.equivalent_radius, 'value', obj.equivalent_radius))
                    
                    tier_candidates.append({
                        "centroid": {
                            "x": round(float(getattr(obj.xcentroid, 'value', obj.xcentroid)), 3), 
                            "y": round(float(getattr(obj.ycentroid, 'value', obj.ycentroid)), 3)
                        },
                        "photometry": {
                            "peak": round(peak, 3), 
                            "flux": round(flux, 3), 
                            "mag": 0.0  # 该底层引擎不提供直接的强度修正
                        }, 
                        "morphology": {
                            "sharpness": 0.0, 
                            "roundness": round(ecc, 3), 
                            "elongation": round(elong, 3),
                            "orientation": round(theta, 3),
                            "equivalent_radius": round(eq_radius, 3)
                        },
                        "meta": {"source": "Engine_A_Macro", "tier": i + 1}
                    })

        # 【聚合坐标处理与 KDTree 二重筛选防交叉网机制】
        tier_added = 0
        
        # 将已完成层的坐标纳入树表比对
        tree = cKDTree(np.array(global_coords)) if global_coords else None
        
        # 缓冲内层的坐标防撞
        current_layer_coords = []

        for cand in tier_candidates:
            cx = cand["centroid"]["x"]
            cy = cand["centroid"]["y"]
            
            # 【截取修正：边际范围遮罩保护 (Margin Masking)】
            margin = 20
            if cx < margin or cx > (W - margin) or cy < margin or cy > (H - margin):
                continue
                
            is_new = True
            cand_pos = np.array([cx, cy])

            # 一查：比对外层空间网格
            if tree is not None:
                dist, _ = tree.query(cand_pos)
                if dist <= 5.0:
                    is_new = False
            
            # 二查：本级的底层计算引擎查重缓存
            if is_new and current_layer_coords:
                dists = np.linalg.norm(np.array(current_layer_coords) - cand_pos, axis=1)
                if np.min(dists) <= 5.0:
                    is_new = False
                    
            if is_new:
                global_stars.append(cand)
                global_coords.append([cx, cy])
                current_layer_coords.append([cx, cy])
                tier_added += 1
                                 
        print(f"      本层探测到全新星点: {tier_added} 颗")

    # ==========================================
    # 模块 4 & 5：分析汇总与渲染输出
    # ==========================================
    print("⏳ 4/5 渲染 UI 界面与生成 JSON 报告...")
    dimmest_star = min(global_stars, key=lambda s: s["photometry"]["peak"]) if global_stars else None

    p_min, p_max = np.percentile(data, (1, 99.9))
    img_8u = (np.clip((data - p_min) / (p_max - p_min) * 255, 0, 255)).astype(np.uint8)
    display_img = cv2.cvtColor(img_8u, cv2.COLOR_GRAY2BGR)

    engine_a_count = 0
    engine_b_count = 0

    # OpenCV 绘图逻辑与计数更新
    for idx, s in enumerate(global_stars):
        cx, cy = int(round(s["centroid"]["x"])), int(round(s["centroid"]["y"]))
        
        if s["meta"]["source"] == "Engine_B_Micro":
            color = (0, 255, 0)
            engine_b_count += 1
        else:
            color = (255, 255, 0)
            engine_a_count += 1
            
        cv2.circle(display_img, (cx, cy), 4, color, 1)
        s["id"] = idx + 1

    # 标记最暗星
    if dimmest_star:
        dx, dy = int(round(dimmest_star["centroid"]["x"])), int(round(dimmest_star["centroid"]["y"]))
        cv2.rectangle(display_img, (dx-12, dy-12), (dx+12, dy+12), (0, 0, 255), 2)
        cv2.putText(display_img, f"Dimmest ID:{dimmest_star['id']}", 
                    (dx + 15, dy), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)

    # HUD 信息面板
    total = len(global_stars)
    cv2.putText(display_img, f"Gradient Stars: {total}", (20, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
    cv2.putText(display_img, f"Macro: {engine_a_count} | Micro: {engine_b_count}", (20, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 1)
    cv2.putText(display_img, f"Gradients: {len(dynamic_gradients)} layers", (20, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 200, 255), 2)

    # 文件输出
    base_name = os.path.splitext(file_name)[0]
    out_img_path = os.path.join(img_out_dir, f"{base_name}.jpg")
    out_json_path = os.path.join(json_out_dir, f"{base_name}.json")
    
    cv2.imwrite(out_img_path, display_img)
    
    with open(out_json_path, 'w', encoding='utf-8') as f:
        json.dump({
            "architecture": "Auto_Calib_Gradient + Double_Engine",
            "stats": {"total_stars": total, "fwhm": round(dynamic_fwhm, 2), "gradients": [round(g, 2) for g in dynamic_gradients]},
            "dimmest_target": dimmest_star,
            "all_stars": global_stars
        }, f, indent=4)

    print(f"✅ 处理完成！共提取: {total} 颗星。")
    print(f"🖼️  结果图已保存至 {img_out_dir} 目录。")
    print(f"📄  结果JSON已保存至 {json_out_dir} 目录。")

if __name__ == "__main__":
    process_star_map_ultimate("/workspace/data/images/img_01.fits", "/workspace/output/single_frames/images", "/workspace/output/single_frames/json", "img_01.fits")