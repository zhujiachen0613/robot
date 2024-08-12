import cv2
import numpy as np
import pybullet as p
import pybullet_data
import time


def find_color_contours(frame, lower, upper):
    # 将图像转换为 HSV 色彩空间
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    # 根据颜色范围创建掩码
    mask = cv2.inRange(hsv, lower, upper)

    # 执行形态学操作来提取对象的轮廓
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    return contours


# 连接到物理引擎
p.connect(p.GUI)
p.setAdditionalSearchPath(pybullet_data.getDataPath())

p.configureDebugVisualizer(p.COV_ENABLE_RENDERING, 1)
p.setRealTimeSimulation(1)  # 启用实时仿真


model_sheet = {
    "plane": "plane.urdf",
    "robot_c": "/home/yuyao/zjc/model/装配体-new.SLDASM/urdf/装配体-new.SLDASM.urdf",
    "robot_c2": "/home/yuyao/zjc/model/装配体123456/urdf/装配体123456.urdf",
    "cylinder": "/home/yuyao/zjc/model/圆柱体/urdf/圆柱体.urdf",
    "cuboid": "/home/yuyao/zjc/model/长方体/urdf/长方体.urdf",
    "pp": "/home/yuyao/zjc/model/pp/urdf/pp.urdf",
    "robot": "/home/yuyao/zjc/model/robot/urdf/robot.urdf",
    "robot2": "/home/yuyao/zjc/model/robot2/urdf/robot.urdf",
}

p.setGravity(0, 0, -9.8)
plane_id = p.loadURDF("plane.urdf")

# 加载机器人
robot_id = p.loadURDF(
    model_sheet["robot"],
    [0, 0, 0],
    useFixedBase=True,
)


joint_indices = range(p.getNumJoints(robot_id))

# 重置机器人模型和其他变量的状态
p.resetBasePositionAndOrientation(robot_id, [0, 0, 0.02], [0, 0, 0, 1])
for i in joint_indices:
    p.resetJointState(robot_id, i, targetValue=0, targetVelocity=0)

#

# 打开摄像头
cap = cv2.VideoCapture("input_video.mp4")

# 定义颜色的 HSV 范围
color_ranges = {
    "yellow": ([30, 150, 150], [45, 255, 255]),
    "brown": ([10, 100, 100], [20, 255, 255]),
    "blue": ([110, 50, 50], [130, 255, 255]),
    "green": ([50, 50, 50], [80, 255, 255]),
}

while True:
    # 读取视频帧
    ret, frame = cap.read()
    if not ret: 
        break

    # 遍历每种颜色
    for color_name, (lower, upper) in color_ranges.items():
        # 寻找特定颜色的轮廓
        contours = find_color_contours(frame, np.array(lower), np.array(upper))

        # 如果找到了轮廓
        if contours:
            # 假设只有一个方块
            cnt = contours[0]
            # 计算方块中心点坐标
            M = cv2.moments(cnt)
            if M["m00"] != 0:
                cx = int(M["m10"] / M["m00"])
                cy = int(M["m01"] / M["m00"])
                print("Color:", color_name, "Center:", (cx, cy))

                # 在这里添加根据识别结果调整机器人关节控制参数的逻辑
                # 例如，可以根据方块的中心坐标调整机器人的末端执行器的位置

    # 显示实时视频
    cv2.imshow("Color Contours", frame)

    # 按下 'q' 键退出循环
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

    time.sleep(0.02)

# 释放资源
cap.release()
cv2.destroyAllWindows()
p.disconnect()
