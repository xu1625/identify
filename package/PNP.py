import numpy as np


def pixels_to_3d(pixel_points, z_values, camera_matrix):
    """
    将像素点+深度转换为3D坐标（相机坐标系）
    :param pixel_points: [(x1,y1), (x2,y2)] 像素坐标
    :param z_values: [z1, z2] 对应点的深度（米）
    :param camera_matrix: [[fx, 0, cx], [0, fy, cy], [0, 0, 1]]
    :return: 3D点列表 [(X1,Y1,Z1), (X2,Y2,Z2)]
    """
    fx, fy = camera_matrix[0, 0], camera_matrix[1, 1]
    cx, cy = camera_matrix[0, 2], camera_matrix[1, 2]

    points_3d = []
    for (x, y), z in zip(pixel_points, z_values):
        X = (x - cx) * z / fx
        Y = (y - cy) * z / fy
        points_3d.append([X, Y, z])

    return np.array(points_3d, dtype=np.float32)

def estimate_yaw(points_3d):
    """
    通过两个3D点的向量估算yaw角（绕Z轴旋转）
    :param points_3d: [(X1,Y1,Z1), (X2,Y2,Z2)]
    :return: yaw（弧度）
    """
    dx = points_3d[1][0] - points_3d[0][0]
    dy = points_3d[1][1] - points_3d[0][1]
    return np.arctan2(dy, dx)  # 返回[-π, π]范围内的角度

def generate_virtual_points(points_3d, armor_width=0.2):
    """
    根据两个实测点和装甲板宽度生成虚拟4点模型
    :param points_3d: 两个实测3D点
    :param armor_width: 装甲板实际宽度（米）
    :return: 4个3D模型点
    """
    # 计算两点向量和垂直方向
    vec = points_3d[1] - points_3d[0]
    length = np.linalg.norm(vec)
    if length < 1e-6:
        raise ValueError("两点距离过近！")

    # 单位向量和法向量（假设装甲板竖直）
    unit_vec = vec / length
    normal_vec = np.array([-unit_vec[1], unit_vec[0], 0])  # Z轴为0

    # 生成4个角点（假设高度为宽度的一半）
    half_width = armor_width / 2
    half_height = half_width / 2

    p1 = points_3d[0] - half_width * unit_vec + half_height * normal_vec
    p2 = points_3d[0] - half_width * unit_vec - half_height * normal_vec
    p3 = points_3d[1] + half_width * unit_vec - half_height * normal_vec
    p4 = points_3d[1] + half_width * unit_vec + half_height * normal_vec

    return np.array([p1, p2, p3, p4], dtype=np.float32)





# # 示例调用
# pixel_points = [(300, 250), (400, 250)]  # YOLO检测的两个点
# z_values = [3.5, 3.5]  # 深度相机提供的Z值（假设相同）
# camera_matrix = np.array([[1000, 0, 640], [0, 1000, 360], [0, 0, 1]])
#
# points_3d = pixels_to_3d(pixel_points, z_values, camera_matrix)
# # 输出：[[-1.02, -0.33, 3.5], [ -0.72, -0.33, 3.5]]


# yaw = estimate_yaw(points_3d)
# print(f"Estimated yaw: {np.degrees(yaw):.1f}°")


# armor_3d_model = generate_virtual_points(points_3d)