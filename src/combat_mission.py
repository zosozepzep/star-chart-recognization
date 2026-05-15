# src/combat_mission.py
import os
import glob
import time
import csv
import logging
import numpy as np
from astropy.io import fits
from astropy.time import Time

# 导入你辛辛苦苦搭建的所有神兵利器
# 注意：请确保你的 main_pipeline.py 和 tracking 文件夹都在 src 目录下
from main_pipeline import StarDetectorPipeline
from tracking.associator import KDTreeAssociator
from tracking.physical_validator import TrackValidator
from benchmark.benchmark_runner import run_detection_benchmark

# 配置实战日志格式
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    datefmt='%H:%M:%S'
)
logger = logging.getLogger("CombatMission")

def run_combat_mission(data_dir: str, output_csv: str, enable_benchmark: bool = False):
    """
    全功率实战入口：支持绝对物理时间排序、GPU/CPU混合加速、卡尔曼时序追踪

    enable_benchmark:
        是否启用 detection-level benchmark。
        如果为 False，只运行原本的 pipeline 和 tracking。
        如果为 True，会在我方 all_raw_detections.csv 生成后自动运行 SExtractor，
        再用 SExtractor catalog 和我方 detection catalog 做 cross-match。
    """
    logger.info("⚔️ INITIATING COMBAT MISSION...")
    
    # ==========================================
    # 阶段 1：绝对时序装填 (Absolute Time Sorter)
    # ==========================================
    search_pattern = os.path.join(data_dir, "*.fit*")
    raw_files = glob.glob(search_pattern)
    
    if not raw_files:
        logger.error(f"No FITS files found in {data_dir}! Aborting.")
        return
        
    logger.info(f" -> Found {len(raw_files)} raw files. Extracting FITS headers...")
    
    time_locked_files = []
    for fp in raw_files:
        try:
            # 极速读取 FITS Header
            header = fits.getheader(fp)
            if 'MJD-OBS' in header:
                obs_time = float(header['MJD-OBS'])
            elif 'DATE-OBS' in header:
                obs_time = Time(header['DATE-OBS']).mjd 
            else:
                obs_time = os.path.getmtime(fp)
            time_locked_files.append((obs_time, fp))
        except Exception as e:
            logger.warning(f"Failed to extract time from {os.path.basename(fp)}: {e}. Skipping.")

    # 按照绝对物理时间排序
    time_locked_files.sort(key=lambda x: x[0])
    fits_files = [fp for _, fp in time_locked_files]
    
    if not fits_files:
        logger.error("No valid FITS files survived the time extraction! Aborting.")
        return

    logger.info(f" -> Chronological lock established. {len(fits_files)} frames ready.")
    logger.info(f" -> First Frame: {os.path.basename(fits_files[0])}")
    logger.info(f" -> Last Frame:  {os.path.basename(fits_files[-1])}")

    # ==========================================
    # 🌟 核心修复：在这里唤醒三栖雷达系统！
    # ==========================================
    logger.info(" -> Booting up Pipeline, Tracker, and Validator...")
    pipeline = StarDetectorPipeline()
    tracker = KDTreeAssociator(match_radius=5.0, max_lost_frames=3)
    
    # 动态生存阈值：要求目标至少在 60% 的帧数中存活
    min_hits_required = max(3, int(len(fits_files) * 0.6))
    validator = TrackValidator(min_hits=min_hits_required, min_r2=0.85)

    t_mission_start = time.time()
# ==========================================
    # 新增：准备 Level 1 原始星点导出管线
    # ==========================================
    raw_catalog_path = os.path.join(os.path.dirname(output_csv), "all_raw_detections.csv")
    logger.info(f"💾 Opening raw detection catalog at {raw_catalog_path}...")
    
    raw_file = open(raw_catalog_path, mode='w', newline='')
    raw_writer = csv.writer(raw_file)
    # 写入表头：加入 Frame_Name 来区分是哪张图的星星
    raw_writer.writerow(["Frame_Name", "X", "Y", "FWHM", "SNR", "Score", "Engine"])
    # ==========================================
    # 阶段 2：逐帧突入 (Sequential Processing)
    # ==========================================
    for frame_idx, file_path in enumerate(fits_files):
        filename = os.path.basename(file_path)
        logger.info(f"\n--- Processing Frame {frame_idx + 1}/{len(fits_files)} : {filename} ---")
        
        try:
            # 读取数据并强制转换为 float64
            data = fits.getdata(file_path)
            data = np.array(data, dtype=np.float64) 
            
            # 【前端：混合加速提取】
            detections = pipeline.run(data)
            # ==========================================
            # 🌟 新增：将当前帧的所有星点无条件保存
            # ==========================================
            for det in detections:
                # 处理可能拟合失败导致的 FWHM 为 NaN 的情况
                fwhm_val = det.fwhm if np.isfinite(det.fwhm) else -1.0
                raw_writer.writerow([
                    filename,  # 记录是哪张 FITS 图
                    f"{det.x:.2f}", 
                    f"{det.y:.2f}", 
                    f"{fwhm_val:.2f}", 
                    f"{det.snr:.1f}", 
                    f"{det.score:.2f}",
                    "+".join(det.engine_votes) # 记录是 DAO 还是 SEG 发现的
                ])
            # 【后端：卡尔曼时序追踪】
            tracker.update(detections, frame_idx)
            logger.info(f" -> Active Tracks in Radar: {len(tracker.active_tracks)}")
            
        except Exception as e:
            logger.error(f"🔥 CRITICAL FAILURE on frame {filename}: {e}")
            logger.warning("Skipping to next frame to maintain radar lock...")
            continue

    # 关闭 Level 1 原始星表文件。
    # benchmark 会读取 all_raw_detections.csv，所以必须先确保数据已经完整写入磁盘。
    raw_file.close()

    # ==========================================
    # 阶段 3：终审与打扫战场 (Validation & Export)
    # ==========================================
    logger.info("\n" + "="*50)
    logger.info("⚖️ ALL FRAMES PROCESSED. INITIATING PHASE 3 VALIDATION...")
    
    raw_confirmed = tracker.get_confirmed_tracks()
    final_pure_tracks = validator.validate_tracks(raw_confirmed)
    
    logger.info(f" -> Raw Confirmed: {len(raw_confirmed)} tracks")
    logger.info(f" -> Pure Targets:  {len(final_pure_tracks)} targets survived")
    
    # 导出 CSV
    logger.info(f"💾 Exporting catalog to {output_csv}...")
    with open(output_csv, mode='w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(["Track_ID", "Type", "Hits", "Start_X", "Start_Y", "End_X", "End_Y", "Velocity_X", "Velocity_Y", "Linearity_R2"])
        
        for trk in final_pure_tracks:
            start_det = trk.detections[0]
            end_det = trk.detections[-1]
            speed = np.hypot(trk.velocity_x, trk.velocity_y)
            
            target_type = "STAR" if speed < 0.5 else "DEBRIS"
            
            writer.writerow([
                trk.track_id,
                target_type,
                trk.hits,
                f"{start_det.x:.2f}", f"{start_det.y:.2f}",
                f"{end_det.x:.2f}", f"{end_det.y:.2f}",
                f"{trk.velocity_x:.3f}", f"{trk.velocity_y:.3f}",
                f"{trk.linearity_r2:.4f}"
            ])

    t_mission_total = time.time() - t_mission_start
    logger.info(f"🏁 COMBAT MISSION ACCOMPLISHED IN {t_mission_total:.2f}s!")

    if enable_benchmark:
        logger.info("📊 BENCHMARK ENABLED. RUNNING SExtractor + cross-match...")
        run_detection_benchmark(
            image_dir=data_dir,
            pipeline_catalog_path=raw_catalog_path,
            output_dir="./output/benchmark",
            match_radius=2.0,
        )
        logger.info("📊 BENCHMARK FINISHED. Results saved to ./output/benchmark")

if __name__ == "__main__":
    # ⚠️ 启动前，请确保这两个路径在你的环境中是正确的
    REAL_DATA_DIRECTORY = "./data/images" # 替换为你的 FITS 文件夹
    OUTPUT_CATALOG = "./output//new/final_catalog.csv"  # 替换为你想要保存 CSV 的路径
    ENABLE_BENCHMARK = True # True 时自动运行 SExtractor 并生成 benchmark 结果
    
    # 如果目录不存在，自动创建输出目录
    os.makedirs(os.path.dirname(OUTPUT_CATALOG), exist_ok=True)
    
    run_combat_mission(
        REAL_DATA_DIRECTORY,
        OUTPUT_CATALOG,
        enable_benchmark=ENABLE_BENCHMARK,
    )
