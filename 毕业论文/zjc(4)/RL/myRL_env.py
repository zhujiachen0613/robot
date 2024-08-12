import math
import time
from typing import Any, SupportsFloat
import pybullet as p
import gymnasium
from gymnasium import spaces
import numpy as np
import pybullet_data
from stable_baselines3 import DQN
import cv2
import csv

model_sheet = {
    "plane": "plane.urdf",
    "robot_c": "/Users/zhujiachen/Desktop/毕业论文/zjc(4)/model/装配体-new.SLDASM/urdf/装配体-new.SLDASM.urdf",
    "robot_c2": "/Users/zhujiachen/Desktop/毕业论文/zjc(4)/model/装配体123456/urdf/装配体123456.urdf",
    "cylinder": "/Users/zhujiachen/Desktop/毕业论文/zjc(4)/model/圆柱体/urdf/圆柱体.urdf",
    "cuboid": "/Users/zhujiachen/Desktop/毕业论文/zjc(4)/model/长方体/urdf/长方体.urdf",
    "pp": "/Users/zhujiachen/Desktop/毕业论文/zjc(4)/model/pp/urdf/pp.urdf",
    "robot": "/Users/zhujiachen/Desktop/毕业论文/zjc(4)/model/robot/urdf/robot.urdf",
    "robot2": "/Users/zhujiachen/Desktop/毕业论文/zjc(4)/model/robot2/urdf/robot.urdf",
}
space_low = 0
space_high = 1
target_useFixedBase = True
action_discount = 10



def cv2_draw_box(frame):
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
        "yellow": ([15, 150, 150], [45, 255, 255]),
        "brown": ([10, 100, 100], [20, 255, 255]),
        "blue": ([110, 50, 50], [130, 255, 255]),
        "green": ([50, 50, 50], [100, 255, 255]),
        "black": ([0, 0, 0], [0, 0, 0]),
    }

    color_rect = {}

    for color_name, (lower, upper) in color_ranges.items():
        # 寻找特定颜色的轮廓
        contours = find_color_contours(frame, np.array(lower), np.array(upper))

        # 计算每个轮廓的面积
        contour_areas = [cv2.contourArea(contour) for contour in contours]
        # 找到最大面积的轮廓索引
        if len(contour_areas) > 0:
            max_area_index = contour_areas.index(max(contour_areas))
            # 获取最大面积的轮廓
            max_area_contour = contours[max_area_index]
            # 找到最小外接矩形
            rect = cv2.minAreaRect(max_area_contour)

            # 获取最小外接矩形
            rect = cv2.minAreaRect(max_area_contour)
            box = cv2.boxPoints(rect)
            box = np.intp(box)

            # 在图像上绘制最小外接矩形
            cv2.drawContours(frame, [box], 0, (0, 0, 255), 1)  # 在绿色通道上绘制矩形，线宽为 2

            (x, y), (w, h), a = rect

            color_rect[color_name] = (x, y, w, h)

            # 绘制轮廓
            # cv2.drawContours(frame, contours, -1, (0, 0, 255), 1)

    if len(color_rect) == 5:
        hand_rects = [
            color_rect["yellow"],
            color_rect["brown"],
            color_rect["blue"],
            color_rect["green"],
        ]
        # hand_rects = color_rect["black"]

        # 计算机器人抓手的中心点
        hand_center_x = int(np.mean([rect[0] + rect[2] // 2 for rect in hand_rects]))
        hand_center_y = int(np.mean([rect[1] + rect[3] // 2 for rect in hand_rects]))

        hand_center_x = np.mean([rect[0] + rect[2]// 2 for rect in [hand_rects[0], hand_rects[3]]])
        hand_center_y = np.mean([rect[1] + rect[3] // 2 for rect in [hand_rects[0], hand_rects[3]]])

        hand_center_x -= 25
        hand_center_y += 10


        cv2.drawContours(
            frame,
            [
                np.intp(
                    [
                        [hand_center_x + 5, hand_center_y + 5],
                        [hand_center_x - 5, hand_center_y - 5],
                        [hand_center_x - 5, hand_center_y + 5],
                        [hand_center_x + 5, hand_center_y - 5],
                    ]
                )
            ],
            0,
            (100, 120, 150),
            1,
        )

    # 显示实时视频
    cv2.imshow("Color Contours", frame)
    # cv2.imwrite("./456.png", frame)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        exit(0)

    return color_rect


# 计算两个矩形之间的距离
def compute_rect_distance(rect1, rect2):
    # 计算矩形1的中心点
    rect1_center_x = rect1[0] + rect1[2] // 2
    rect1_center_y = rect1[1] + rect1[3] // 2

    # 计算矩形2的中心点
    rect2_center_x = rect2[0] + rect2[2] // 2
    rect2_center_y = rect2[1] + rect2[3] // 2

    # 计算两个矩形中心点之间的距离
    distance_x = abs(rect1_center_x - rect2_center_x)
    distance_y = abs(rect1_center_y - rect2_center_y)

    # 计算两个矩形中心点在x和y方向上的距离之和
    total_distance = distance_x + distance_y
    return total_distance


# 计算距离权重的函数
def compute_distance_weight(hand_rects, block_rect):
    # 计算机器人抓手的中心点

    hand_center_x = np.mean([rect[0] + rect[2] // 2 for rect in hand_rects])
    hand_center_y = np.mean([rect[1] + rect[3] // 2 for rect in hand_rects])

    hand_center_x = np.mean([rect[0] + rect[2] // 2 for rect in [hand_rects[0], hand_rects[3]]])
    hand_center_y = np.mean([rect[1] + rect[3] // 2 for rect in [hand_rects[0], hand_rects[3]]])
    # 计算机器人抓手中心与方块中心之间的距离
    hand_center_x -= 25
    hand_center_y += 10

    distance = compute_rect_distance((hand_center_x, hand_center_y, 0, 0), block_rect)
    hand_distance_to_block_rect = 0
    for rect in hand_rects:
        hand_distance_to_block_rect += compute_rect_distance(rect,block_rect)
    hand_distance_to_block_rect = compute_rect_distance(hand_rects[0], block_rect)

    # 根据距离计算权重
    max_distance = 500.0  # 最大距离（可根据实际情况调整）
    max_hand_distace = 500
    weight = 1.0 - min(distance / max_distance, 1.0)  # 距离越近，权重越高
    weight_hand = 1.0 - min(hand_distance_to_block_rect / max_hand_distace, 1.0)
    return weight,weight_hand


class DiscretizedActionWrapper(gymnasium.ActionWrapper):
    def __init__(self, env, bins):
        super(DiscretizedActionWrapper, self).__init__(env)
        self.bins = bins
        # 创建离散动作空间
        self.action_space = spaces.Discrete(bins)
        self.low = env.action_space.low
        self.high = env.action_space.high
        self.env = env

    def action(self, action):
        # 将离散动作转换为连续动作
        # print(action)
        # action = np.linspace(self.low, self.high, self.bins)[action]
        self.env.action_space.seed(int(action))
        action = self.env.action_space.sample()
        # print(action)
        return action

    def reverse_action(self, action):
        # 将连续动作转换为离散动作
        # print(action)
        bin_width = (self.high - self.low) / (self.bins - 1)
        action_discrete = np.round((action - self.low) / bin_width).astype(int)
        # print(action_discrete)
        return action_discrete


def process_action(action, state):
    new_action = []
    for i in action.tolist():
        match i:
            case i if i <= -0.66 and i >= -1:
                new_action.append(-1.0 / action_discount)
            case i if i < 0.66 and i > -0.66:
                new_action.append(0.0 / action_discount)
            case i if i <= 1 and i >= 0.66:
                new_action.append(1.0 / action_discount)

    return state + np.array(new_action), np.array(new_action)


class RobotEnv(gymnasium.Env):
    """
    强化学习的环境
    """

    def __init__(self):
        super(RobotEnv, self).__init__()
        #初始化PyBullet物理引擎
        self._physics_client_id = p.connect(p.GUI)  # 或者使用 p.DIRECT 进行无界面模式
        #self._physics_client_id = p.connect(p.DIRECT)  # 或者使用 p.DIRECT 进行无界面模式
        # 创建机器人模型并设置初始状态
        self.current_distance = None

        self.camera_real = False

        self._config_pybullet()
        self._config_camera()
        self._config_space()
        self._count = 0
        self.read_action = []
        self.danci_final_action = []
        self.ww = []
        self.w1 = 0
        self.w2 = 0
        self.w0_9 = True
        self.w0_8 = True

        self.previous_distance = 0
        self.previous_distance_hand = 0

        self.count_reward = 0.0
        self.danci_reward = []

    def _config_pybullet(self):
        """
        pybullet 初始化配置
        """

        # <---------- 关闭图形渲染,在没有加载模型的情况下，就不初始化渲染 ----------> #
        p.configureDebugVisualizer(p.COV_ENABLE_RENDERING, 0)
        p.setAdditionalSearchPath(pybullet_data.getDataPath())

        # <---------- 设置环境重力加速度 ----------> #
        p.setGravity(0, 0, -0.98)

        # <---------- 加载模型 ----------> #
        planeId = p.loadURDF(model_sheet["plane"])
        self.robot_id = p.loadURDF(
            model_sheet["robot_c"],
            [0, 0, 0],
            useFixedBase=True,
        )
        self.target_id = p.loadURDF(
            model_sheet["cuboid"],
            [0.1, -0.4, 0],
            [0, 0, 45, 0],
            useFixedBase=target_useFixedBase,
            useMaximalCoordinates=False,
        )

        # <---------- 开启图形渲染 ----------> #
        p.configureDebugVisualizer(p.COV_ENABLE_RENDERING, 1)
        p.setRealTimeSimulation(1)  # 启用实时仿真

        self.joint_indices = range(p.getNumJoints(self.robot_id))



    def _config_camera(self):
        # 摄像机
        self.view_matrix = None
        self.projection_matrix = None
        self.camera_distance = 0.6
        self.camera_yaw = 180
        self.camera_pitch = -89
        self.camera_target_position = [0, -0.35, 0]  # 摄像机的目标位置
        self.width = 500
        self.height = 480

        # <---------- 设置摄像机 ----------> #
        p.resetDebugVisualizerCamera(
            self.camera_distance,
            self.camera_yaw,
            self.camera_pitch,
            self.camera_target_position,
        )  # 拍照片
        self.view_matrix = p.computeViewMatrixFromYawPitchRoll(
            self.camera_target_position,
            self.camera_distance,
            self.camera_yaw,
            self.camera_pitch,
            0,
            2,
        )
        self.projection_matrix = p.computeProjectionMatrixFOV(
            fov=60, aspect=float(self.width) / self.height, nearVal=0.1, farVal=100
        )

    def _get_image_camera(self):
        return (
            self._get_image_camera_real()
            if self.camera_real
            else self._get_image_camera_pybullet()
        )

    def _get_image_camera_pybullet(self):
        """
        增加摄像头获取俯视角图片
        """

        (_, _, px, _, _) = p.getCameraImage(
            self.width, self.height, self.view_matrix, self.projection_matrix
        )  # 如果你想使用硬件加速的OpenGL渲染器，可以将renderer参数设置为p.ER_BULLET_HARDWARE_OPENGL
        image = cv2.cvtColor(px, cv2.COLOR_RGB2BGR)
        # 保存图像文件
        # cv2.imwrite("./123.png", image)
        return image

    def _get_image_camera_real(self):
        # TODO 增加现实使用cv videoCapture获取图形的image(真实环境：需要切换到此)
        pass

    def _config_space(self):
        self.num_joints = p.getNumJoints(self.robot_id)
        print(self.num_joints)

        # TODO  定义动作空间和状态空间
        self.action_space = spaces.Box(
            low=space_low, high=space_high, shape=(self.num_joints,), dtype=np.float32
        )
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(self.num_joints,), dtype=np.float32
        )
        self.step_num = 0

    def reset(self, seed=None, options=None):
        # if hasattr(self, "last_action"):
        #     delattr(self, "last_action")
        # 重置机器人模型和其他变量的状态
        p.resetBasePositionAndOrientation(self.robot_id, [0, 0, 0], [0, 0, 0, 1])
        for i in range(self.num_joints):
            p.resetJointState(self.robot_id, i, targetValue=0, targetVelocity=0)

        p.resetBasePositionAndOrientation(
            self.target_id, [0.075, -0.138, 0.00001], [0, 0, 0, 1]
        )
        self.w0_9 = True
        self.w0_8 = True

        self.previous_distance = 0
        self.previous_distance_hand = 0

        # 获取初始观测状态
        observation = self._get_observation()
        image = self._get_image_camera()
        self._count = 0

        self.count_reward = 0.0
        return observation, image

    def step(self, action):
        # 执行给定的动作，控制机器人模型
        if self._count == 0:
            self._apply_action(np.array([0,0,0,0]))
            p.stepSimulation(physicsClientId=self._physics_client_id)
            time.sleep(0.3)

        action, new_action = process_action(action, self._get_observation())
        self.read_action.append(new_action)
        print("第", self._count, "轮")
        print(action,new_action)
        self._apply_action(action)
        self._count += 1
        if self._count > 25:
            self.danci_final_action.append(action)
            self.ww.append([self.w1, self.w2])
            self.danci_reward.append([self.count_reward])
            self.reset()

        # 模拟仿真一步
        p.stepSimulation(physicsClientId=self._physics_client_id)
        time.sleep(0.2)
        # 获取下一个观测状态
        next_observation = self._get_observation()
        # 计算奖励
        reward, distance = self._calculate_reward()
        # 判断是否完成任务
        done = self._check_if_done(distance)
        # 返回下一个状态、奖励、完成状态和其他信息
        return next_observation, reward, done, None, {}

    def _get_observation(self):
        if not hasattr(self, "robot"):
            assert Exception("robot hasn't been loaded in!")

        # 获取机器人当前的关节角度作为观测状态
        observation = []
        for i in range(self.num_joints):
            joint_info = p.getJointState(self.robot_id, i)
            observation.append(joint_info[0])

        return np.array(observation)

    def _calculate_reward(self):
        img = self._get_image_camera()

        color_rect = cv2_draw_box(img)

        if len(color_rect) == 5:
            weight , weight_hand = compute_distance_weight(
                [
                    color_rect["yellow"],
                    color_rect["brown"],
                    color_rect["green"],
                    color_rect["blue"],
                ],
                color_rect["black"],
            )
            print(weight,"   ",weight_hand)

            self.w1 = weight
            self.w2 = weight_hand
            # 计算距离变化
            if weight > self.previous_distance:
                reward1 = (weight - self.previous_distance)*100
                self.previous_distance = weight
            else:
                reward1 = 0

            if weight_hand > self.previous_distance_hand:
                reward2 = (weight_hand - self.previous_distance_hand)*100
                self.previous_distance_hand = weight_hand
            else:
                reward2 = 0

            current_distance = weight
            #distance_change = self.previous_distance - current_distance
            #self.previous_distance = current_distance

            current_distance_hand = weight_hand
            #distance_change_hand = self.previous_distance_hand - current_distance_hand
            #self.previous_distance_hand = current_distance_hand

        else:
            self.reset()
            return -1, -1000

        """
        # 如果距离缩小，给予正奖励；如果距离增大，给予负奖励
        if distance_change >= 0.1:
            reward1 = -30  #
        elif distance_change < 0.1 and distance_change >= 0:
            reward1 = -10  # 负奖励
        elif distance_change <= -0.1:
            reward1 = 10
        #elif distance_change > -0.1 and  distance_change < 0:
        else:
            reward1 = 3  # 负奖励
        #else:
        #   reward1 = -2  # 保持不变，无奖励
        
        if distance_change_hand <= -0.1:
            reward2 = 10 # 正奖励
        elif distance_change_hand < 0 and distance_change_hand > -0.1:
            reward2 = 2  # 负奖励
        elif distance_change_hand >= 0.1:
            reward2 = -20 # 正奖励
        #elif distance_change_hand > 0 and distance_change_hand < 0.1:
        else:
            reward2 = -5  # 负奖励
        #else:
        #    reward2 = -2  # 保持不变，无奖励
        if weight > 0.9 and self.w0_9 == True:
            reward1 = reward1 + 200
            self.w0_9 = False
        if weight_hand > 0.8 and self.w0_8 == True:
            reward2 = reward2 + 200
            self.w0_8 = False
        """
        self.count_reward += (reward1 + reward2)

        return reward1 + reward2, current_distance



    def _check_if_done(self, distance):
        """
        检查是否完成任务
        """
        target_distance = 1

        # TODO 如果4个距离都在数值之内,就结束
        self.step_num += 1
        if distance > target_distance or self.step_num > 36000000:
            done = True
        else:
            done = False
        return done

    def _apply_action(self, action):
        # TODO 设定具体动作
        p.setJointMotorControlArray(
            self.robot_id,
            range(self.num_joints),
            p.POSITION_CONTROL,
            targetPositions=action,
        )

    def render(self, mode="human"):
        # 可视化机器人模型和环境状态
        pass

    def close(self):
        # 关闭环境并清理资源
        print(self.danci_reward)
        print("ggg")
        with open("danci_final_action_50000.csv", "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerows(self.danci_final_action)
        with open("danci_reward_50000.csv", "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerows(self.danci_reward)
        #with open("danci_reward_50000.csv", "w", newline="") as f:
        #    writer = csv.writer(f)
        #    writer.writerows(self.danci_reward)

        with open("weight_hand_weight_50000.csv", "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerows(self.ww)
        if self._physics_client_id >= 0:
            p.disconnect()
        self._physics_client_id = -1

    def seed(self, seed=None):
        self.np_random, seed = gymnasium.utils.seeding.np_random(seed)
        return [seed]


if __name__ == "__main__":
    env = RobotEnv()
    observation = env.reset()
    for t in range(100000000):
        action = env.action_space.sample()  # 示例：随机采样动作
        # action = np.random.random(4)
        # time.sleep(0.1)  # 控制循环速度
        next_observation, reward, done, _, _ = env.step(action)
        if done:
            break

        observation = next_observation

    env.close()
