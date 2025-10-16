import cv2
from ultralytics import YOLO
import argparse
import time
import numpy as np
import track


# # 正确导入你的自定义模块
# from ekf import ExtendedKalmanFilter
# from track import Track

# # 导入需要的函数（如果这些函数在track.py中）
# from track import pixels_to_3d, estimate_yaw_from_two_points, normalize_angle


def out_dectation(x, y):
    """回调函数"""
    if x is not None and y is not None:
        return x, y
    else:
        print('回调函数：未检测到目标')
        return None, None


def run_realtime_detection(camera_index=0, choose_team=0, callback=None):
    # 最佳模型
    model = YOLO('run/train4/weights/best.pt')
    # 打开摄像头
    capture = cv2.VideoCapture(camera_index)
    # 设置全屏窗口
    cv2.namedWindow("Fullscreen", cv2.WND_PROP_FULLSCREEN)
    cv2.setWindowProperty("Fullscreen", cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)

    if not capture.isOpened():
        print('打不开当前摄像头')
        return

    prev_time = time.time()
    Track = track.Track()  # 创建Track实例

    # 相机内参
    camera_matrix = np.array([
        [1000, 0, 640],
        [0, 1000, 360],
        [0, 0, 1]
    ])

    try:
        while capture.isOpened():
            ret, frame = capture.read()
            if not ret:
                break

            # 计算时间间隔
            curr_time = time.time()
            dt = max(0.001, curr_time - prev_time)
            fps = 1.0 / dt
            prev_time = curr_time

            # YOLO检测
            results = model.predict(
                source=frame,
                conf=0.25,
                iou=0.45,
                stream=True,
                verbose=False
            )

            annotated_frame = frame.copy()
            detections = []  # 存储检测结果

            for result in results:
                annotated_frame = result.plot()  # 绘制检测结果

                if result.boxes is not None:
                    boxes = result.boxes.xyxy.cpu().numpy()
                    confidences = result.boxes.conf.cpu().numpy()
                    classes = result.boxes.cls.cpu().numpy()

                    for box, conf, cls in zip(boxes, confidences, classes):
                        cls_id = int(cls)
                        if (choose_team == 0 and cls_id == 6) or (choose_team == 1 and cls_id == 7):
                            x1, y1, x2, y2 = map(int, box)

                            # 计算yaw角
                            pixel_points = [(x1, y1), (x2, y2)]
                            z_values = [2.0, 2.0]  # 假设深度，需要改进

                            try:
                                points_3d = track.pixels_to_3d(pixel_points, z_values, camera_matrix)
                                yaw = track.estimate_yaw_from_two_points(points_3d)
                                yaw = track.normalize_angle(yaw)

                                # 格式: [x1, y1, x2, y2, z, yaw]
                                detections.append([x1, y1, x2, y2, z_values[0], yaw])
                            except Exception as e:
                                print(f"计算yaw角错误: {e}")
                                continue

            # 显示FPS和信息
            cv2.putText(annotated_frame, f"FPS: {fps:.1f}", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            cv2.putText(annotated_frame, f"Detections: {len(detections)}", (10, 60),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

            # 更新跟踪器
            if detections:
                # 调用track的update方法
                targets = Track.update(detections, dt)
                aims=[]

                # 绘制跟踪结果
                for i, target_state in enumerate(targets):
                    # 从状态向量中提取信息
                    x, y, yaw, w, h = target_state[0], target_state[2], target_state[6], target_state[9], target_state[10]
                    x_1, y_1 = int(x - w / 2), int(y - h / 2)
                    x_2, y_2 = int(x + w / 2), int(y + h / 2)

                    # 绘制跟踪框
                    cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), (255, 255, 255), 4)
                    cv2.putText(annotated_frame, f"Track {i}", (x1, y1 - 10),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
                    x=(x_1+x_2)/2
                    y=(y_1+y_2)/2
                    aims.append([x,y,yaw])
                target_sub_list = max(aims, key=lambda sub: sub[2])

                # 3. 提取对应的x和y
                target_x = target_sub_list[0]
                target_y = target_sub_list[1]
                # target_yaw = target_sub_list[2]


                if callback:
                    callback(target_x, target_y)

            # 显示图像
            cv2.imshow('Fullscreen', annotated_frame)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    finally:
        capture.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--camera', help='选择摄像头', default=0, type=int)
    parser.add_argument('--team', help='选择我方队伍，0为红方，1为蓝方', default=0, type=int)
    args = parser.parse_args()

    run_realtime_detection(args.camera, args.team, callback=out_dectation)