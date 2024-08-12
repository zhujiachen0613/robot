import time
import cv2
import numpy as np


def find_color_contours(frame, lower, upper):
    # 将图像转换为 HSV 色彩空间
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    # 根据颜色范围创建掩码
    mask = cv2.inRange(hsv, lower, upper)

    # 执行形态学操作来提取对象的轮廓
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    return contours


# 定义颜色的 HSV 范围
color_ranges = {
    "yellow": ([30, 150, 150], [45, 255, 255]),
    "brown": ([10, 100, 100], [20, 255, 255]),
    "blue": ([110, 50, 50], [130, 255, 255]),
    "green": ([50, 50, 50], [80, 255, 255]),
}

# 打开摄像头
cap = cv2.VideoCapture("input_video.mp4")

while True:
    # 读取视频帧
    ret, frame = cap.read()
    if not ret:
        break

    # 遍历每种颜色
    for color_name, (lower, upper) in color_ranges.items():
        # 寻找特定颜色的轮廓
        contours = find_color_contours(frame, np.array(lower), np.array(upper))

        # 绘制轮廓
        cv2.drawContours(frame, contours, -1, (0, 0, 255), 2)

    # 显示实时视频
    cv2.imshow("Color Contours", frame)

    # 按下 'q' 键退出循环
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

    time.sleep(0.02)

# 释放资源
cap.release()
cv2.destroyAllWindows()
