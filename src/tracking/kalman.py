# src/tracking/kalman.py
import numpy as np

class ConstantVelocityKalmanFilter:
    def __init__(self, init_x: float, init_y: float):
        """初始化四维状态空间与协方差矩阵"""
        # 状态向量 [x, y, vx, vy]^T
        self.x = np.array([init_x, init_y, 0.0, 0.0], dtype=np.float64)

        # 状态转移矩阵 F (假设两帧之间 dt = 1)
        self.F = np.array([
            [1, 0, 1, 0],
            [0, 1, 0, 1],
            [0, 0, 1, 0],
            [0, 0, 0, 1]
        ], dtype=np.float64)

        # 观测矩阵 H (我们只能通过图像测量到 x 和 y，无法直接测量速度)
        self.H = np.array([
            [1, 0, 0, 0],
            [0, 1, 0, 0]
        ], dtype=np.float64)

        # 误差协方差矩阵 P (表示系统对当前状态的信心，初始时速度极度不确定)
        self.P = np.array([
            [2.0, 0, 0, 0],   
            [0, 2.0, 0, 0],
            [0, 0, 10.0, 0],  # 速度初始方差极大
            [0, 0, 0, 10.0]
        ], dtype=np.float64)

        # 过程噪声矩阵 Q (容忍目标进行微小的非匀速机动)
        q = 0.1
        self.Q = np.array([
            [q, 0, 0, 0],
            [0, q, 0, 0],
            [0, 0, q*5, 0],
            [0, 0, 0, q*5]
        ], dtype=np.float64)

        self.I = np.eye(4)

    def predict(self) -> tuple[float, float]:
        """执行先验预测，返回预测的 (x, y)"""
        self.x = self.F @ self.x
        self.P = self.F @ self.P @ self.F.T + self.Q
        return self.x[0], self.x[1]

    def update(self, meas_x: float, meas_y: float, snr: float):
        """执行后验修正：这是一个极其优雅的动态权重更新"""
        z = np.array([meas_x, meas_y], dtype=np.float64)

        # 🌟 杀手级功能：动态观测噪声！
        # 如果信噪比极低（如 SNR=3），r_val 变大，系统更信任物理惯性预测
        # 如果信噪比极高（如 SNR=100），r_val 趋近 0.1，系统死死咬住图像观测值
        r_val = max(0.1, 10.0 / (snr + 1e-5))
        R = np.array([
            [r_val, 0],
            [0, r_val]
        ], dtype=np.float64)

        # 1. 计算创新向量 (Innovation)
        y = z - self.H @ self.x
        
        # 2. 计算卡尔曼增益 (Kalman Gain)
        S = self.H @ self.P @ self.H.T + R
        K = self.P @ self.H.T @ np.linalg.inv(S)

        # 3. 状态与协方差更新
        self.x = self.x + K @ y
        self.P = (self.I - K @ self.H) @ self.P