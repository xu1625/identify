import numpy as np
from scipy.linalg import inv


class ExtendedKalmanFilter:
    """
    扩展卡尔曼滤波器(EKF)完整实现
    针对装甲板跟踪场景设计（状态维度11，观测维度6）
    """

    def __init__(self):
        # ----------------- 状态维度 -----------------
        self.state_dim = 11  # [x,vx,y,vy,z,vz,yaw,vyaw,r,w,h]
        self.meas_dim = 6  # [x,y,z,yaw,w,h]

        # ----------------- 初始化状态和协方差 -----------------
        self.x = np.zeros(self.state_dim)
        self.P = np.eye(self.state_dim)

        # ----------------- 噪声矩阵 -----------------
        self.Q = np.diag([0.1, 0.5, 0.1, 0.5, 0.1, 0.5, 0.2, 0.3, 0.01, 0.1, 0.1])  # 过程噪声
        self.R = np.diag([10, 10, 5, 0.5, 5, 5])  # 观测噪声 (单位：像素/弧度)

    def predict(self, dt):
        """
        非线性状态预测
        :param dt: 时间步长(秒)
        """
        # ---- 1. 非线性状态转移 ----
        self.x = self._state_transition(self.x, dt)

        # ---- 2. 计算状态转移雅可比 ----
        F = self._jacobian_F(self.x, dt)

        # ---- 3. 更新协方差 ----
        self.P = F @ self.P @ F.T + self.Q

    def update(self, z):
        """
        非线性观测更新
        :param z: 观测向量 [x,y,z,yaw,w,h]
        """
        # ---- 1. 非线性观测预测 ----
        z_pred = self._observation_model(self.x)

        # ---- 2. 计算观测雅可比 ----
        H = self._jacobian_H(self.x)

        # ---- 3. 计算卡尔曼增益 ----
        S = H @ self.P @ H.T + self.R
        K = self.P @ H.T @ inv(S)

        # ---- 4. 状态更新 ----
        self.x += K @ (z - z_pred)
        self.P = (np.eye(self.state_dim) - K @ H) @ self.P

    # ------------ 非线性模型定义 ------------
    def _state_transition(self, x, dt):
        """状态转移函数 f(x,u)"""
        new_x = x.copy()
        # 位置更新 (非线性部分：yaw角影响)
        new_x[0] += x[1] * dt   #* np.cos(x[6])  # x += vx*cos(yaw)*dt
        new_x[2] += x[1] * dt   #* np.sin(x[6])  # y += vx*sin(yaw)*dt
        new_x[4] += x[5] * dt  # z += vz*dt
        new_x[6] += x[7] * dt  # yaw += vyaw*dt
        return new_x

    def _observation_model(self, x):
        """观测函数 h(x)"""
        return np.array([
            x[0],  # x
            x[2],  # y
            x[4],  # z
            self._normalize_angle(x[6]),  # yaw (归一化到[-π,π])
            x[9],  # w
            x[10]  # h
        ])

    def _jacobian_F(self, x, dt):

        """状态转移的雅可比矩阵"""
        F = np.eye(self.state_dim)

        # 位置对速度的偏导
        F[0, 1] = dt  # ∂x/∂vx
        F[3, 4] = dt  # ∂y/∂vy
        F[6, 7] = dt  # ∂z/∂vz

        # 位置对加速度的偏导
        F[0, 2] = 0.5 * dt ** 2  # ∂x/∂ax
        F[3, 5] = 0.5 * dt ** 2  # ∂y/∂ay

        # 速度对加速度的偏导
        F[1, 2] = dt  # ∂vx/∂ax
        F[4, 5] = dt  # ∂vy/∂ay

        # 角度对角速度的偏导
        F[8, 9] = dt  # ∂yaw/∂vyaw

        return F
        # """状态转移雅可比 ∂f/∂x"""
        # F = np.eye(self.state_dim)
        # yaw = x[6]
        # # 位置对速度的偏导
        # F[0, 1] = dt * np.cos(yaw)
        # F[2, 1] = dt * np.sin(yaw)
        # # 位置对yaw的偏导 (非线性关键!)
        # F[0, 6] = -x[1] * dt * np.sin(yaw)
        # F[2, 6] = x[1] * dt * np.cos(yaw)
        # # yaw对vyaw的偏导
        # F[6, 7] = dt
        # return F

    def _jacobian_H(self, x):
        """观测雅可比 ∂h/∂x"""
        H = np.zeros((self.meas_dim, self.state_dim))
        H[0, 0] = 1  # ∂x/∂x
        H[1, 2] = 1  # ∂y/∂y
        H[2, 4] = 1  # ∂z/∂z
        H[3, 6] = 1  # ∂yaw/∂yaw (注意：实际需考虑归一化的影响)
        H[4, 9] = 1  # ∂w/∂w
        H[5, 10] = 1  # ∂h/∂h
        return H

    @staticmethod
    def _normalize_angle(angle):
        """将角度归一化到[-π, π]"""
        while angle > np.pi:
            angle -= 2 * np.pi
        while angle < -np.pi:
            angle += 2 * np.pi
        return angle


# ===================== 使用示例 =====================
if __name__ == "__main__":
    # 初始化EKF
    ekf = ExtendedKalmanFilter()

    # 设置初始状态 (示例值)
    ekf.x = np.array([100, 1.5, 200, 0, 1.2, 0, 0.1, 0.05, 0.2, 50, 50])

    # 模拟预测-更新循环
    for t in range(10):
        dt = 0.1  # 100ms

        # ---- 预测 ----
        ekf.predict(dt)
        print(f"Time {t:.1f}s - Predicted state: {ekf.x[:4]}...")

        # ---- 模拟观测 ----
        if t % 2 == 0:  # 每2步收到一次观测
            z = np.array([100 + t * 15, 200 + t * 5, 1.2, 0.1 + t * 0.05, 50, 50])
            ekf.update(z)
            print(f"Updated with measurement: {z}")
            print(f"Posterior covariance trace: {np.trace(ekf.P):.2f}")


# import numpy as np
#
#
#
#
# class ExtendedKalmanFilter:
#     def __init__(self, state_dim, meas_dim):
#         # 状态维度（例如11维：x,y,z,vx,vy,vz,yaw,vyaw,r,w,h）
#         self.state_dim = state_dim
#         # 测量维度（例如6维：x,y,z,yaw,w,h）
#         self.meas_dim = meas_dim
#
#         # 状态向量和协方差矩阵
#         self.x = np.zeros(state_dim)
#         self.P = np.eye(state_dim)
#
#         # 过程噪声和观测噪声
#         self.Q = np.eye(state_dim)
#         self.R = np.eye(meas_dim)
#
#         # 非线性函数和雅可比矩阵
#         self.f = None  # 状态转移函数
#         self.h = None  # 观测函数
#         self.JF = None  # 状态转移雅可比
#         self.JH = None  # 观测雅可比
#
#     def predict(self, dt):
#         # 非线性状态预测
#         self.x = self.f(self.x, dt)
#
#         # 更新协方差
#         F = self.JF(self.x, dt)
#         self.P = F @ self.P @ F.T + self.Q
#
#     def update(self, z):
#         # 非线性观测预测
#         z_pred = self.h(self.x)
#
#         # 计算卡尔曼增益
#         H = self.JH(self.x)
#         S = H @ self.P @ H.T + self.R
#         K = self.P @ H.T @ np.linalg.inv(S)
#
#         # 状态更新
#         self.x += K @ (z - z_pred)
#         self.P = (np.eye(self.state_dim) - K @ H) @ self.P
#
# # 状态转移函数（非线性）
# def f(x, dt):
#     # x: [xc, vx, yc, vy, z, vz, yaw, vyaw, r, w, h]
#     new_x = x.copy()
#     new_x[0] += x[1] * dt  # xc += vx*dt
#     new_x[2] += x[3] * dt  # yc += vy*dt
#     new_x[4] += x[5] * dt  # z += vz*dt
#     new_x[6] += x[7] * dt  # yaw += vyaw*dt
#     return new_x
#
# # 观测函数（非线性）
# def h(x):
#     return np.array([x[0], x[2], x[4], x[6], x[9], x[10]])  # [xc, yc, z, yaw, w, h]
#
# # 雅可比矩阵计算
# def jacobian_F(x, dt):
#     F = np.eye(11)
#     F[0, 1] = dt
#     F[2, 3] = dt
#     F[4, 5] = dt
#     F[6, 7] = dt
#     return F
#
# def jacobian_H(x):
#     H = np.zeros((6, 11))
#     H[0, 0] = 1  # dxc/dxc
#     H[1, 2] = 1  # dyc/dyc
#     H[2, 4] = 1  # dz/dz
#     H[3, 6] = 1  # dyaw/dyaw
#     H[4, 9] = 1  # dw/dw
#     H[5, 10] = 1  # dh/dh
#     return H