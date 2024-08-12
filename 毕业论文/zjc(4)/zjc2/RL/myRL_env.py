import pybullet as p
import gym
from gym import spaces
import numpy as np
import pybullet_data
from stable_baselines3 import DQN


class RobotEnv(gym.Env):

    def __init__(self):
        super(RobotEnv, self).__init__()
        #初始化PyBullet物理引擎
        #self._physics_client_id = p.connect(p.GUI)  # 或者使用 p.DIRECT 进行无界面模式
        self._physics_client_id = p.connect(p.DIRECT)  # 或者使用 p.DIRECT 进行无界面模式
        # 创建机器人模型并设置初始状态
        self.robot_id = None
        self.target_id = None

        self.current_distance = None
        self._config_init()
        self.num_joints = p.getNumJoints(self.robot_id)

        #TODO  定义动作空间和状态空间
        self.action_space = spaces.Box(low=-10,
                                       high=10,
                                       shape=(self.num_joints, ),
                                       dtype=np.float32)
        
        self.observation_space = spaces.Box(low=-np.inf,
                                            high=np.inf,
                                            shape=(self.num_joints, ),
                                            dtype=np.float32)
        self.step_num = 0

    def _config_init(self):
        """
            pybullet 初始化配置
        """

        # 渲染逻辑,在没有加载模型的情况下，就不初始化渲染
        p.configureDebugVisualizer(p.COV_ENABLE_RENDERING, 0)
        p.setAdditionalSearchPath(pybullet_data.getDataPath())

        # 设置环境重力加速度
        p.setGravity(0, 0, -0.98)

        # 加载模型
        planeId = p.loadURDF("plane.urdf")
        self.robot_id = p.loadURDF("/home/hp/zjc/robot/urdf/robot.urdf",
                                   [0, 0, 0],
                                   useFixedBase=True)
        self.target_id = p.loadURDF("/home/hp/zjc/pp/urdf/pp.urdf", [1, 0, 0],
                                    useFixedBase=False,
                                    useMaximalCoordinates=False)
        #开启图形渲染
        p.configureDebugVisualizer(p.COV_ENABLE_RENDERING, 1)

        p.setRealTimeSimulation(1)  #启用实时仿真

    def reset(self):
        # 重置机器人模型和其他变量的状态
        p.resetBasePositionAndOrientation(self.robot_id, [0, 0, 0],
                                          [0, 0, 0, 1])
        for i in range(self.num_joints):
            p.resetJointState(self.robot_id,
                              i,
                              targetValue=0,
                              targetVelocity=0)

        # 获取初始观测状态
        observation = self._get_observation()

        return observation

    def step(self, action):
        
        # 执行给定的动作，控制机器人模型
        self._apply_action(action)

        # 模拟仿真一步
        p.stepSimulation(physicsClientId=self._physics_client_id)

        # 获取下一个观测状态
        next_observation = self._get_observation()

        # 计算奖励
        reward, distance = self._calculate_reward()

        # 判断是否完成任务
        done = self._check_if_done(distance)

        # 返回下一个状态、奖励、完成状态和其他信息
        return next_observation, reward, done, {}

    def render(self, mode='human'):
        # 可视化机器人模型和环境状态
        pass

    def close(self):
        # 关闭环境并清理资源
        if self._physics_client_id >= 0:
            p.disconnect()
        self._physics_client_id = -1

    def seed(self, seed=None):
        self.np_random, seed = gym.utils.seeding.np_random(seed)
        return [seed]

    def _get_observation(self):
        if not hasattr(self, "robot"):
            assert Exception("robot hasn't been loaded in!")

        # 获取机器人当前的关节角度作为观测状态
        observation = []
        for i in range(self.num_joints):
            joint_info = p.getJointState(self.robot_id, i)
            observation.append(joint_info[0])

        #TODO 增加摄像头获取俯视角图片
        return np.array(observation)

    def _calculate_reward(self):
        """
            计算奖励        
        """
        # robot_pos, _ = p.getBasePositionAndOrientation(self.robot_id)
        # object_pos, _ = p.getBasePositionAndOrientation(self.target_id)
        # distance = np.linalg.norm(np.array(robot_pos) - np.array(object_pos)
        #TODO 使用opencv检测物体和被抓物体的距离
        
        #TODO 写一个计算正负奖励的算法，距离如果缩小，给予正奖励。如果距离增大，给与负奖励
        if distance < previous_distance:
            reward = 1
        else:
            reward = -1

        return reward, distance

    def _get_square(self) -> list:
        """
            计算 检测方形的位置
        """
        square_indices = [[1, 2, 3, 4], [6, 7, 8, 9], [11, 12, 13, 14],
                          [16, 17, 18, 19]]
        square_positions = []

        for square in square_indices:
            # 获取关节的位置和姿态
            joint_states = p.getJointStates(self.robot_id, square)
            # 计算正方形的位置
            square_position = np.zeros(3)
            for joint_state in joint_states:
                link_state = p.getLinkState(self.robot_id, int(joint_state[0]))
                link_position = link_state[0]
                square_position += np.array(link_position)

            square_position /= len(joint_states)
            square_positions.append(square_position)

        # 打印每个正方形的位置
        for i, square_position in enumerate(square_positions):
            print("Square", i + 1, "position:", square_position)
        return square_positions

    def _check_if_done(self, distance):
        """
            检查是否完成任务
        """

        #TODO 如果4个距离都在数值之内,就结束
        self.step_num += 1
        if distance < target_distance or self.step_num > 36000000:
            done = True
        else:
            done = False
        return done

    def _apply_action(self, action):
        #TODO 设定具体动作
        p.setJointMotorControlArray(self.robot_id,
                                    range(self.num_joints),
                                    p.POSITION_CONTROL,
                                    targetPositions=action)


if __name__ == '__main__':
    env = RobotEnv()
    observation = env.reset()
    for t in range(100000000):
        action = env.action_space.sample()  # 示例：随机采样动作
        next_observation, reward, done, _ = env.step(action)

        if done:
            break

        observation = next_observation

    env.close()
