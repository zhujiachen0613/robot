import pybullet as p
import pybullet_data
import time
import math

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


# 控制机器人的关节
for _ in range(10000000):
    # 控制每个关节
    for i, joint_index in enumerate(joint_indices):
        # 在这里添加您的控制逻辑
        # 这是一个简单的示例，每个关节都做简单的周期性运动
        target_position = math.sin(_ / 100) * 0.5  # 使用正弦函数生成周期性的目标位置
        p.setJointMotorControl2(
            robot_id, joint_index, p.POSITION_CONTROL, targetPosition=target_position
        )

    p.stepSimulation()
    time.sleep(1.0 / 240.0)  # 控制循环速度

# 断开连接
p.disconnect()
