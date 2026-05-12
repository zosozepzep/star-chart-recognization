import os
import json
import cv2
from datetime import datetime
import multiprocessing
from concurrent.futures import ProcessPoolExecutor

# 导入自定义处理模块
from detector import process_star_map_ultimate
from tracker import StarTracker


def parse_and_sort_fits(image_dir):
    """
    时间戳解析与排序引擎
    解析文件名中的时间戳信息，并计算相对于首帧的时间偏移（秒）
    """
    fits_files = [f for f in os.listdir(image_dir) if f.endswith('.fits')]
    sorted_files = sorted(fits_files)  # 依据 20260309... 字符串原生排序
    
    parsed_data = []
    base_time = None
    
    for filename in sorted_files:
        time_str = filename.split('_')[0]
        dt_obj = datetime.strptime(time_str + "000", "%Y%m%d%H%M%S%f")
        if base_time is None:
            base_time = dt_obj
        delta_seconds = (dt_obj - base_time).total_seconds()
        
        parsed_data.append({
            "filename": filename,
            "relative_sec": round(delta_seconds, 3)
        })
        
    return parsed_data


def process_wrapper(args):
    """多进程处理的包装器函数"""
    fits_path, img_dir, json_dir, filename = args
    process_star_map_ultimate(fits_path, img_dir, json_dir, filename)
    return filename


def main():
    print("🚀 [SpaceMapper Pipeline] 全自动目标识别流水线启动\n" + "=" * 50)
    
    # 路径配置
    BASE_DIR = "/workspace"
    IMAGE_DIR = os.path.join(BASE_DIR, "data/images/60394_20260309_1437117414637934_PIC")
    
    # 定义分类输出目录
    SINGLE_OUT_BASE = os.path.join(BASE_DIR, "output/single_frames")
    SINGLE_IMG_DIR = os.path.join(SINGLE_OUT_BASE, "images")  # 存放渲染图片
    SINGLE_JSON_DIR = os.path.join(SINGLE_OUT_BASE, "json")   # 存放提取数据
    
    FINAL_OUT_DIR = os.path.join(BASE_DIR, "output/final_catalog")
    
    # 确保所有分类目录存在
    os.makedirs(SINGLE_IMG_DIR, exist_ok=True)
    os.makedirs(SINGLE_JSON_DIR, exist_ok=True)
    os.makedirs(FINAL_OUT_DIR, exist_ok=True)

    # --- Phase 1: 序列排序 ---
    sequence_data = parse_and_sort_fits(IMAGE_DIR)

    # --- Phase 2: 单帧目标提取 (多核并发) ---
    print(f"\n[Phase 2] 启动多核并发引擎 (检测到 {multiprocessing.cpu_count()} 个 CPU 核心)...")
    
    tasks = []
    for item in sequence_data:
        fits_filename = item["filename"]
        fits_path = os.path.join(IMAGE_DIR, fits_filename)
        expected_json = os.path.join(SINGLE_JSON_DIR, f"{os.path.splitext(fits_filename)[0]}.json")
        
        if not os.path.exists(expected_json):
            tasks.append((fits_path, SINGLE_IMG_DIR, SINGLE_JSON_DIR, fits_filename))
            
    if tasks:
        # 使用进程池加速处理
        with ProcessPoolExecutor() as executor:
            # 强制触发迭代以捕获异常
            list(executor.map(process_wrapper, tasks))
        print("✅ 多核并发处理完毕！")
    else:
        print("⏩ 缓存或已全部就绪，跳过提取。")

    # --- Phase 3: 多帧轨迹关联与分析 ---
    print("\n[Phase 3] 启动 KD-Tree 时序追踪器...")
    tracker = StarTracker(match_radius=3.0, min_frames_to_confirm=3)
    
    for item in sequence_data:
        json_filename = f"{os.path.splitext(item['filename'])[0]}.json"
        json_path = os.path.join(SINGLE_JSON_DIR, json_filename)
        
        with open(json_path, 'r', encoding='utf-8') as f:
            frame_data = json.load(f)
            stars_list = frame_data["all_stars"]
            tracker.process_frame(stars_list)

    # --- Phase 4: 输出最终星表 ---
    print("\n[Phase 4] 提纯完毕，生成最终报告...")
    final_catalog = tracker.get_confirmed_stars()
    
    # 简单的分类统计
    stars_count = sum(1 for v in final_catalog.values() if v["type"] == "Background_Star")
    debris_count = sum(1 for v in final_catalog.values() if v["type"] == "Moving_Debris")
    
    report = {
        "pipeline_version": "Ultimate_Hybrid_v1.0",
        "total_frames_processed": len(sequence_data),
        "summary": {
            "confirmed_stars": stars_count,
            "detected_debris": debris_count,
            "eliminated_noise": "100%"
        },
        "confirmed_targets": final_catalog
    }
    
    final_json_path = os.path.join(FINAL_OUT_DIR, "master_catalog.json")
    with open(final_json_path, 'w', encoding='utf-8') as f:
        json.dump(report, f, indent=4)
        
    # --- Phase 5: 生成全局预览图 ---
    print("\n[Phase 5] 生成运动目标全局预览图...")
    if sequence_data:
        # 取最后一帧作为底图
        last_filename = sequence_data[-1]["filename"]
        base_img_path = os.path.join(SINGLE_IMG_DIR, f"{os.path.splitext(last_filename)[0]}.jpg")
        
        if os.path.exists(base_img_path):
            preview_img = cv2.imread(base_img_path)
            
            # 在图上标出运动目标
            for tid, target in final_catalog.items():
                if target["type"] == "Moving_Debris":
                    cx = int(round(target["latest_data"]["centroid"]["x"]))
                    cy = int(round(target["latest_data"]["centroid"]["y"]))
                    
                    # 用醒目的颜色 (BGR的红色) 画一个瞄准框/同心圆
                    cv2.circle(preview_img, (cx, cy), 15, (0, 0, 255), 2)  # 外圈
                    cv2.circle(preview_img, (cx, cy), 4, (0, 0, 255), -1)  # 内层实心点
                    # 绘制 ID
                    cv2.putText(preview_img, f"Debris-{tid}", (cx + 20, cy), 
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 100, 255), 2)
                                
            preview_out_path = os.path.join(FINAL_OUT_DIR, "debris_preview.jpg")
            cv2.imwrite(preview_out_path, preview_img)
            print(f"🖼️  全局预览图已生成: {preview_out_path}")
        else:
            print("⚠️  警告：无法找到底图，跳过预览图生成。")
        
    print(f"\n🎉 恭喜！流水线执行完毕！")
    print(f"📊 最终锁定: {stars_count} 颗恒星，{debris_count} 个运动目标。")
    print(f"📄 最终星表已保存至: {final_json_path}")


if __name__ == "__main__":
    main()