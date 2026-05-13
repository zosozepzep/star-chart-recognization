# src/main_pipeline.py
import logging
import time
import numpy as np
# 导入检测流水线模块
from detector.candidate_generator import CandidateGenerator
from detector.feature_extractor import FeatureExtractor
from detector.psf_fitter import PSFFitter
from detector.scorer import SourceScorer
from detector.nms import spatial_nms
from tracking.associator import KDTreeAssociator
from tracking.physical_validator import TrackValidator
from detector.gpu_preprocessor import HybridBackgroundEstimator
# 设定工程化日志标准
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-7s | %(name)s | %(message)s",
    datefmt="%H:%M:%S"
)
logger = logging.getLogger("StarPipeline")

class StarDetectorPipeline:
    def __init__(self):
        """初始化流水线的各个节点 (Node)"""
        logger.info("Initializing Pipeline Modules...")
        # 实例化混合架构 GPU 预处理器
        self.preprocessor = HybridBackgroundEstimator(box_size=64, sigma_clip=3.0)
        self.generator = CandidateGenerator(
            dao_threshold_sigma=3.0, 
            seg_threshold_sigma=2.5, 
            merge_radius=3.0
        )
        self.extractor = FeatureExtractor(patch_size=15, aperture_radius=3.0)
        self.fitter = PSFFitter(patch_size=11)
        self.scorer = SourceScorer(acceptance_threshold=0.60) # 设置及格线
        self.associator = KDTreeAssociator()
        self.validator = TrackValidator()
    def run(self, data: np.ndarray):
        """执行端到端检测"""
        t0 = time.time()
        
        # 1. 候选生成 (高召回率)
        logger.info("Step 1: Generating candidates...")
        candidates = self.generator.generate(data)
        logger.info(f" -> Found {len(candidates)} rough candidates.")
        if not candidates:
            return []

        # 2. 基础特征提取 (测光与背景估计)
        logger.info("Step 2: Extracting photometric features...")
        candidates = self.extractor.extract(data, candidates)

        # 3. PSF 拟合 (亚像素定位与形态学建模)
        logger.info("Step 3: Fitting 2D Gaussian PSF...")
        candidates = self.fitter.fit_sources(data, candidates)
        
        success_fits = sum(1 for c in candidates if c.fit_success)
        logger.info(f" -> PSF Fit Success: {success_fits}/{len(candidates)}")

        # 4. 统一置信度打分
        logger.info("Step 4: Scoring sources...")
        candidates = self.scorer.score_sources(candidates)

        # 5. 过滤与空间非极大值抑制 (NMS)
        logger.info("Step 5: Filtering and Spatial NMS...")
        accepted = [c for c in candidates if self.scorer.accepted(c)]
        final_sources = spatial_nms(accepted, radius=3.0)
        
        t_total = time.time() - t0
        logger.info(f"Pipeline finished in {t_total:.3f}s. Final targets: {len(final_sources)}")
        
        return final_sources

def generate_synthetic_starfield() -> np.ndarray:
    """生成带有高斯白噪声和已知坐标的单帧模拟星空 (1024x1024)"""
    logger.info("Generating synthetic single-frame starfield...")
    shape = (1024, 1024)
    data = np.random.normal(loc=100.0, scale=5.0, size=shape)
    
    true_stars = [
        (100.2, 150.8, 500),   # 极亮星
        (500.5, 500.5, 200),   # 中等亮星
        (800.1, 200.9, 50),    # 暗弱星
        (900.0, 900.0, 20),    # 极限暗星
        (10.5, 10.5, 300)      # 边缘星
    ]
    
    y, x = np.mgrid[:shape[0], :shape[1]]
    sigma = 1.5
    for x0, y0, amp in true_stars:
        star = amp * np.exp(-((x - x0)**2 + (y - y0)**2) / (2 * sigma**2))
        data += star
        
    return data

def generate_moving_sequence(frames=15) -> list[np.ndarray]:
    """生成包含匀速运动目标和随机噪声的多帧时序测试数据"""
    logger.info(f"Generating synthetic sequence of {frames} frames...")
    shape = (1024, 1024)
    sequence = []
    
    # 目标的物理运动方程
    debris_x_start, debris_y_start = 100.0, 100.0
    debris_vx, debris_vy = 2.5, 1.2 # 每帧位移
    debris_amp = 80 # 暗弱碎片
    
    static_stars = [(500.5, 500.5, 200), (800.1, 200.9, 150)]
    sigma = 1.5
    y, x = np.mgrid[:shape[0], :shape[1]]
    
    for i in range(frames):
        data = np.random.normal(loc=100.0, scale=5.0, size=shape)
        
        for sx, sy, samp in static_stars:
            data += samp * np.exp(-((x - sx)**2 + (y - sy)**2) / (2 * sigma**2))
            
        current_x = debris_x_start + debris_vx * i
        current_y = debris_y_start + debris_vy * i
        data += debris_amp * np.exp(-((x - current_x)**2 + (y - current_y)**2) / (2 * sigma**2))
        
        sequence.append(data)
        
    return sequence

# ==========================================
# 自动化执行流 (Runners)
# ==========================================
def test_single_frame_pipeline(pipeline: StarDetectorPipeline):
    """运行单帧特征提取与评分测试"""
    print("\n" + "="*50)
    print("🔬 RUNNING SINGLE-FRAME TEST")
    print("="*50)
    
    test_data = generate_synthetic_starfield()
    final_results = pipeline.run(test_data)
    
    print(f"\n{'Engine':<10} | {'X':<8} | {'Y':<8} | {'FWHM':<6} | {'SNR':<7} | {'Score':<5}")
    print("-" * 55)
    for res in sorted(final_results, key=lambda x: x.score, reverse=True):
        fwhm_val = res.fwhm if np.isfinite(res.fwhm) else -1
        print(f"{'+'.join(res.engine_votes):<10} | {res.x:>8.2f} | {res.y:>8.2f} | "
              f"{fwhm_val:>6.2f} | {res.snr:>7.1f} | {res.score:>5.2f}")

def test_multi_frame_tracking(pipeline: StarDetectorPipeline):
    """运行多帧卡尔曼滤波与时序关联测试"""
    print("\n" + "="*50)
    print("🚀 RUNNING MULTI-FRAME TRACKING TEST (KALMAN + KD-TREE)")
    print("="*50)
    
    sequence = generate_moving_sequence(frames=15)
    tracker = KDTreeAssociator(match_radius=4.0, min_hits_to_confirm=3)
    
    t_start = time.time()
    for frame_idx, frame_data in enumerate(sequence):
        # A. 单帧提取高置信度目标
        final_sources = pipeline.run(frame_data)
        
        # B. 喂给卡尔曼追踪引擎进行时序关联
        tracker.update(final_sources, frame_idx)
        
        active_cnt = len(tracker.active_tracks)
        confirmed_cnt = len(tracker.get_confirmed_tracks())
        logger.info(f"[Frame {frame_idx:02d}] Active Tracks: {active_cnt} | CONFIRMED: {confirmed_cnt}")

    t_total = time.time() - t_start
    print("\n" + "-"*50)
    print(f"🏁 TRACKING COMPLETE in {t_total:.2f}s")
    print("-" * 50)
    print("Initiating Phase 3: Physical Validation...")
    # 实例化审查：要求至少存活 5 帧，线性度 R^2 至少 0.85
    validator = TrackValidator(min_hits=5, min_r2=0.85)
    
    # 提取 Phase 2 给出的初步确认轨迹
    raw_confirmed_tracks = tracker.get_confirmed_tracks()
    
    # 执行终审
    final_pure_tracks = validator.validate_tracks(raw_confirmed_tracks)
    
    print(f" -> Before Validation: {len(raw_confirmed_tracks)} tracks")
    print(f" -> After Validation:  {len(final_pure_tracks)} pure targets")
    print("-" * 50)
    for trk in final_pure_tracks:
        start_det = trk.detections[0]
        end_det = trk.detections[-1]
        
        speed = np.hypot(trk.velocity_x, trk.velocity_y)
        target_type = "⭐ Static Star" if speed < 0.5 else "🛰️ MOVING DEBRIS"
        
        print(f"Track ID: {trk.track_id:03d} | {target_type}")
        print(f"  -> Hits: {trk.hits}/15 frames")
        print(f"  -> Start: ({start_det.x:.1f}, {start_det.y:.1f})")
        print(f"  -> End:   ({end_det.x:.1f}, {end_det.y:.1f})")
        print(f"  -> Kalman Velocity: [vx={trk.velocity_x:.2f}, vy={trk.velocity_y:.2f}] px/frame")
        print("-" * 50)

if __name__ == "__main__":
    # 实例化一个全局复用的 Pipeline
    shared_pipeline = StarDetectorPipeline()
    
    # 通过注释/取消注释来选择运行哪个测试，或者一起运行
    #test_single_frame_pipeline(shared_pipeline)
    test_multi_frame_tracking(shared_pipeline)