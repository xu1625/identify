import numpy as np
import ekf
from Filter.matcher import Matcher

#yaw
def pixels_to_3d(pixel_points, z_values, camera_matrix):
    """将像素坐标+深度转换为3D坐标（相机坐标系）"""
    fx, fy = camera_matrix[0, 0], camera_matrix[1, 1]
    cx, cy = camera_matrix[0, 2], camera_matrix[1, 2]

    points_3d = []
    for (x, y), z in zip(pixel_points, z_values):
        X = (x - cx) * z / fx
        Y = (y - cy) * z / fy
        points_3d.append([X, Y, z])
    return np.array(points_3d, dtype=np.float32)

def estimate_yaw_from_two_points(points_3d):
    """通过两个3D点计算yaw角（绕Z轴旋转）"""
    dx = points_3d[1, 0] - points_3d[0, 0]  # X方向差值
    dy = points_3d[1, 1] - points_3d[0, 1]  # Y方向差值
    return np.arctan2(dy, dx)  # 返回[-π, π]范围内的弧度值

def normalize_angle(angle):
    """将角度规范化到[-π, π]范围内"""
    while angle > np.pi:
        angle -= 2 * np.pi
    while angle < -np.pi:
        angle += 2 * np.pi
    return angle

class Track:
    def __init__(self):
        # self.track_id = 0  # 目标ID
        self.state_list = []  # 存储EKF实例
        # self.track_colors = {}  # 存储每个目标的绘制颜色

    def update(self, detections, dt):
        """
        处理单帧数据
        :param detections: 检测结果列表 [[x1,y1,x2,y2,z,yaw], ...]
        :param dt: 时间步长
        :return: 当前有效目标状态列表
        """
        # ---- 1. 预测 ----
        for ekf_instance in self.state_list:
            ekf_instance.predict(dt)

        # ---- 2. 数据关联 ----
        mea_list = [self._detection_to_meas(d) for d in detections]
        matches, unmatched_tracks, unmatched_dets = self._associate(self.state_list, mea_list)

        # ---- 3. 更新匹配目标 ----
        for track_idx, meas_idx in matches:
            self.state_list[track_idx].update(mea_list[meas_idx])

        # ---- 4. 处理未匹配的跟踪器 ----
        self.state_list = [ekf_inst for i, ekf_inst in enumerate(self.state_list)
                          if i not in unmatched_tracks]

        # ---- 5. 初始化新目标 ----
        for idx in unmatched_dets:
            new_ekf = ekf.ExtendedKalmanFilter()
            new_ekf.x = self._meas_to_state(mea_list[idx])
            self.state_list.append(new_ekf)
            # self.track_colors[len(self.state_list)-1] = np.random.randint(0, 255, 3)

        return [ekf_inst.x for ekf_inst in self.state_list]

    @staticmethod
    def _detection_to_meas(det):
        """将检测框转为EKF观测格式 [x,y,z,yaw,w,h]"""
        x1, y1, x2, y2, z, yaw = det
        return np.array([(x1+x2)/2, (y1+y2)/2, z, yaw, abs(x2-x1), abs(y2-y1)])

    @staticmethod
    def _meas_to_state(meas):
        """将观测值转为状态向量 [x,vx,y,vy,z,vz,yaw,vyaw,r,w,h]"""
        return np.array([meas[0], 0, meas[1], 0, meas[2], 0, meas[3], 0, 0.2, meas[4], meas[5]])


    def _associate(self, kalman_list, mea_list):
        """数据关联方法"""
        if not kalman_list or not mea_list:
            return [], list(range(len(kalman_list))), list(range(len(mea_list)))

        # 将状态类进行转换便于统一匹配类型
        state_list = []  # [x, y, z, yaw, w, h].T
        for kalman in kalman_list:
            state = kalman.x  # 使用预测后的状态
            state_list.append(np.array([
                state[0],  # x
                state[2],  # y
                state[4],  # z
                state[6],  # yaw
                state[9],  # w
                state[10]  # h
            ]))

        # 进行匹配
        match_dict = Matcher.match(state_list, mea_list)

        # 解析匹配结果 - 只返回匹配信息，不进行更新
        matches = []
        state_used = set()
        mea_used = set()

        for state_key, mea_key in match_dict.items():
            try:
                state_index = int(state_key.split('_')[1])
                mea_index = int(mea_key.split('_')[1])
                if state_index < len(kalman_list) and mea_index < len(mea_list):
                    matches.append([state_index, mea_index])  # ✅ 只记录匹配，不更新
                    state_used.add(state_index)
                    mea_used.add(mea_index)
            except (ValueError, IndexError) as e:
                print(f"解析匹配结果错误: {e}")
                continue

        # 求出未匹配状态和量测
        unmatched_tracks = [i for i in range(len(kalman_list)) if i not in state_used]
        unmatched_dets = [i for i in range(len(mea_list)) if i not in mea_used]

        return matches, unmatched_tracks, unmatched_dets