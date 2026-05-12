import numpy as np
from scipy.spatial import cKDTree


class StarTracker:
    def __init__(self, match_radius=3.0, min_frames_to_confirm=3):
        """
        多帧星点追踪引擎
        :param match_radius: 两帧之间同一个目标允许的最大位移偏差 (像素)
        :param min_frames_to_confirm: 连续出现多少帧才被确认为真星/真实目标
        """
        self.match_radius = match_radius
        self.min_frames_to_confirm = min_frames_to_confirm
        
        # 内部状态记录
        self.active_tracks = {}  # 正在追踪的轨迹 {track_id: [历史星点数据列表]}
        self.next_track_id = 0
        self.current_frame_idx = 0

    def process_frame(self, stars_in_current_frame):
        """
        输入当前帧提取出的所有星点数据 (JSON 字典的列表)，进行轨迹关联
        """
        self.current_frame_idx += 1
        
        # 第一帧：直接初始化所有星点为新的独立轨迹
        if not self.active_tracks:
            for star in stars_in_current_frame:
                self.active_tracks[self.next_track_id] = [star]
                self.next_track_id += 1
            return
            
        # 提取上一帧还在追踪的所有坐标点，用于建树
        prev_ids = list(self.active_tracks.keys())
        
        # 取每个轨迹列表里最后（最新）的一个坐标
        prev_coords = np.array([
            [self.active_tracks[tid][-1]["centroid"]["x"], 
             self.active_tracks[tid][-1]["centroid"]["y"]] 
            for tid in prev_ids
        ])
        
        tree = cKDTree(prev_coords)
        
        # 遍历当前帧的每一颗星
        for star in stars_in_current_frame:
            curr_pos = [star["centroid"]["x"], star["centroid"]["y"]]
            
            # 寻找上一帧中距离最近的目标
            dist, idx = tree.query(curr_pos)
            
            # 如果距离小于阈值，说明是同一个天体
            if dist <= self.match_radius:
                track_id = prev_ids[idx]
                self.active_tracks[track_id].append(star)
            else:
                # 如果周围没有匹配的历史点，说明这是一个新出现的目标（或者噪点）
                self.active_tracks[self.next_track_id] = [star]
                self.next_track_id += 1
                
        # TODO (可选): 清理那些在当前帧“消失”的历史轨迹，防止内存泄漏
        # 对于航天短时序(如15帧)，也可以保留，看它后续帧会不会重新出现
        
    def get_confirmed_stars(self):
        """
        清洗噪点：只返回连续出现次数达标的真实天体，并引入牛顿运动学终审
        """
        confirmed = {}
        
        for tid, track in self.active_tracks.items():
            if len(track) >= self.min_frames_to_confirm:
                # 提取轨迹中所有的坐标点
                points = np.array([[pt["centroid"]["x"], pt["centroid"]["y"]] for pt in track])
                displacement = np.linalg.norm(points[-1] - points[0])
                
                # ==========================================
                # 【运动学审核】
                # ==========================================
                if displacement < 5.0:
                    # 类别 A：总位移极小，是在原地微小抖动的【背景恒星】
                    target_type = "Background_Star"
                else:
                    # 类别 B：产生了显著位移，疑似碎片，必须进行“线性度”体检
                    # 向量化计算所有相邻帧点之间的距离和，即实际轨迹总长
                    trajectory_length = np.sum(np.linalg.norm(points[1:] - points[:-1], axis=1))
                    
                    # 计算线性度 (理想直线为 1.0)
                    linearity = displacement / trajectory_length if trajectory_length > 0 else 0
                    
                    if linearity > 0.95: 
                        # 真正的物理直线运动
                        target_type = "Moving_Debris"
                    else:
                        # 不规则碎片，判断为噪点
                        target_type = "False_Positive_Noise"
                # ==========================================

                # 核心拦截：如果是噪点，直接抛弃，绝不写入最终星表
                if target_type == "False_Positive_Noise":
                    continue

                # 只有真星和真碎片才能获得身份认证
                confirmed[tid] = {
                    "type": target_type,
                    "appearances": len(track),
                    "displacement_px": round(displacement, 2),
                    "latest_data": track[-1]  # 返回最新一帧的物理信息
                }
                
        return confirmed