import cv2
import numpy as np
import random

# 定义视频的宽度和高度
width = 900
height = 900
# 定义视频的帧率
fps = 30
# 定义视频时长（秒）
duration = 30
# 计算视频总共有多少帧
total_frames = int(fps * duration)

# 创建VideoWriter对象，使用MP4编解码器
out = cv2.VideoWriter(
    "input_video.mp4", cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height)
)

# 初始化方块的位置
square_size = 100
square_padding = 100
square_centers = [
    (width - square_size - square_padding, int(height / 4) * (i + 0.5))
    for i in range(4)
]

# 定义方块的移动范围
move_range = 10

# 循环生成每一帧
for frame_count in range(total_frames):
    # 创建空白画布
    frame = np.zeros((height, width, 3), dtype=np.uint8)

    # 绘制四个彩色方块
    colors = [(0, 255, 255), (50, 69, 19), (255, 0, 0), (0, 255, 0)]
    # colors = [(0, 255, 255), (139, 69, 19), (255, 0, 0), (0, 255, 0)]
    for i, color in enumerate(colors):
        x, y = square_centers[i]

        # # 随机微调方块的位置
        # dx = random.randint(-move_range, move_range)
        # dy = random.randint(-move_range, move_range)
        # new_x = x + dx
        # new_y = y + dy

        # 计算目标位置
        target_x = x + random.randint(-move_range, move_range)
        target_y = y + random.randint(-move_range, move_range)

        # 计算新位置（使用线性插值）
        alpha = min(1, frame_count / total_frames)  # 插值系数
        new_x = x + alpha * (target_x - x)
        new_y = y + alpha * (target_y - y)

        # 检查新位置是否超出边界或者与其他方块重叠
        while (
            new_x < 0
            or new_x + square_size > width
            or new_y < 0
            or new_y + square_size > height
            or any(
                (
                    new_x + square_size > other_x
                    and new_x < other_x + square_size
                    and new_y + square_size > other_y
                    and new_y < other_y + square_size
                )
                for other_x, other_y in square_centers
                if (other_x, other_y) != (x, y)
            )
        ):
            # dx = random.randint(-move_range, move_range)
            # dy = random.randint(-move_range, move_range)
            target_x = x + random.randint(-move_range, move_range)
            target_y = y + random.randint(-move_range, move_range)
            new_x = x + alpha * (target_x - x)
            new_y = y + alpha * (target_y - y)
            # new_x = x + dx
            # new_y = y + dy

        square_centers[i] = (new_x, new_y)
        cv2.rectangle(
            frame,
            (int(new_x), int(new_y)),
            (int(new_x) + square_size, int(new_y) + square_size),
            color,
            -1,
        )

    # 绘制黑色圆形
    circle_radius = 25
    circle_center = (int(width / 4), int(height / 2))
    cv2.circle(frame, circle_center, circle_radius, (0, 0, 0), -1)

    # 写入视频帧
    out.write(frame)

# 释放资源
out.release()
