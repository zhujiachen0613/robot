"""
模型加载和预定配置
"""

import math
import time
import pybullet_data
import pybullet as p


class PyBulletModel:
    """
    Documentation for PyBulletModel.

    Attributes:
        - p : pybullet对象
        - physicsClientId : 仿真环境对象
        - robot_id : 机器人对象
        - robot_model_path : urdf模型路径
        - self.object_id : 被抓取物体对象
        - object_model_path : 被抓取模型路径
        - startPos : 模型位置
        - startOrientation : 模型方向

    Methods:
        - print_robot_location():  获取模型位置与方向四元数
        - print_robot_info(): 获取模型本身信息
        - get_use_joints(): 获取可用的关节
        - set_camera(): 控制相机
        - set_Joint(): 控制关节
        - add_reset_btn(): 增加重置按钮
        - reset_init(): 判断按钮重置模型

    """

    def __init__(self,
                 robot_model_path,
                 use_gui,
                 startPos,
                 startOrientation,
                 object_model_path=None) -> None:
        self.p = p
        self.robot_id = None
        self.object_id = None
        self.physicsClientId = None
        self.robot_model_path = robot_model_path
        self.object_model_path = object_model_path
        self.startPos = startPos
        self.startOrientation = self.p.getQuaternionFromEuler(startOrientation)

        self.physicsCilent_connect(use_gui)  #初始化连接
        self.init_config()  # 初始化配置

    def physicsCilent_connect(self, use_gui):
        """
            连接物理引擎

            Parameters : 
                - use_gui : pybullet.GUI 显示界面  pybullet.DIRECT 不显示
        """
        if use_gui == "y":
            self.physicsClientId = self.p.connect(self.p.GUI)
        else:
            self.physicsClientId = self.p.connect(self.p.DIRECT)

    def stepSimulation(self):
        self.p.stepSimulation()

    def init_config(self):
        """
            pybullet 初始化配置
        """

        # 渲染逻辑,在没有加载模型的情况下，就不初始化渲染
        self.p.configureDebugVisualizer(self.p.COV_ENABLE_RENDERING, 0)

        # p.configureDebugVisualizer(p.COV_ENABLE_GUI, 0) #取消多余界面
        # p.configureDebugVisualizer(p.COV_ENABLE_TINY_RENDERER, 0)#禁用调试可视化器的微型渲染器
        # p.configureDebugVisualizer(p.COV_ENABLE_RENDERING, 0)#禁用调试可视化器的渲染功能

        # 添加资源路径,凡是在pybullet_data这个文件夹下的模型，我们都可以直接使用它们的文件名加载
        self.p.setAdditionalSearchPath(pybullet_data.getDataPath())

        # 设置环境重力加速度
        self.p.setGravity(0, 0, -0.98)

        # 加载模型
        self.load_robot()

        #开启图形渲染
        self.p.configureDebugVisualizer(self.p.COV_ENABLE_RENDERING, 1)

        self.p.setRealTimeSimulation(1)  #启用实时仿真

    def add_text(self, text, textPosition, textColorRGB, textSize):
        """
            创建文字

            Parameters : 
                - text : 文本
                - textPosition : 文本位置
                - textColorRGB : 文本颜色
                - textSize : 文本大小

            Returns:
                -  text_id : 文本对象

            example : text="Destination", textPosition=[0, 1, 3], textColorRGB=[0, 1, 0], textSize=1.2
        """
        text_id = self.p.addUserDebugText(text=text,
                                          textPosition=textPosition,
                                          textColorRGB=textColorRGB,
                                          textSize=textSize)
        return text_id

    def load_robot(self):
        """
            加载机器模型、修改模型属性
        """
        # 加载URDF模型，此处是加载蓝白相间的陆地
        # stadiumId = self.p.loadSDF("stadium.sdf") #草坪
        planeId = self.p.loadURDF("plane.urdf")

        # 加载机器人，并设置加载的机器人的位姿
        self.robot_id = self.p.loadURDF(self.robot_model_path,
                                        self.startPos,
                                        self.startOrientation,
                                        useFixedBase=False,
                                        useMaximalCoordinates=False)

        if self.object_model_path:
            self.object_id = self.p.loadURDF(self.object_model_path,
                                             self.startPos,
                                             self.startOrientation,
                                             useFixedBase=False,
                                             useMaximalCoordinates=False)

        # # 创建固定约束将机器人与地面连接起来
        # constraint_id = self.p.createConstraint(planeId, -1, self.robot_id, -1,
        #                                         p.JOINT_FIXED, [0, 0, 0],
        #                                         [0, 0, 0], [0, 0, 0])

        # # 设置约束的最大力
        # self.p.changeConstraint(constraint_id, maxForce=100000)

        # 设置模型的姿态为躺平
        # self.p.resetBasePositionAndOrientation(self.robot_id, startPos, [0, 0, 0, 1])

        # 修改模型的物理属性
        # mass = 1.0  # 新的质量
        # friction = 0.5  # 新的摩擦系数
        # inertia = [0.1, 0.1, 0.1]  # 新的惯性

        # self.p.changeDynamics(self.robot_id, -1, mass=mass)
        # self.p.changeDynamics(self.robot_id, -1, lateralFriction=friction)
        # self.p.changeDynamics(self.robot_id, -1, localInertiaDiagonal=inertia)

    def get_use_joints(self):
        """
            获取可用的关节

            Returns(dict):
                - available_joints(list) : 机器人可用关节名称
                - available_joints_indexes(list) : 机器人可用关节索引
        """
        available_joints_indexes = [
            i for i in range(self.p.getNumJoints(self.robot_id))
            if self.p.getJointInfo(self.robot_id, i)[2] != self.p.JOINT_FIXED
        ]
        available_joints = [
            self.p.getJointInfo(self.robot_id, i)[1]
            for i in available_joints_indexes
        ]
        print(available_joints)
        return {
            "available_joints": available_joints,
            "available_joints_indexes": available_joints_indexes
        }

    def get_robot_location(self):
        """
            获取模型位置与方向四元数

            Returns(dict):
                - cubePos : 机器人的位置坐标
                - cubeOrn : 机器人的朝向四元数
        """

        cubePos, cubeOrn = self.p.getBasePositionAndOrientation(self.robot_id)
        print("-" * 20)
        print(f"机器人的位置坐标为:{cubePos}\n机器人的朝向四元数为:{cubeOrn}")
        print("-" * 20)

        return {"cubePos": cubePos, "cubeOrn": cubeOrn}

    def get_robot_info(self):
        """
            获取模型本身信息

            Returns(dict):
                - info_tuple_list(list): [info_tuple1,info_tuple2...]
                    - 关节序号 info_tuple[0]
                    - 关节名称 info_tuple[1]
                    - 关节类型 info_tuple[2]
                    - 机器人第一个位置的变量索引 info_tuple[3]
                    - 机器人第一个速度的变量索引 info_tuple[4]
                    - 保留参数 info_tuple[5]
                    - 关节的阻尼大小  info_tuple[6]
                    - 关节的摩擦系数 info_tuple[7]
                    - slider和revolute(hinge)类型的位移最小值 info_tuple[8]
                    - slider和revolute(hinge)类型的位移最大值 info_tuple[9]
                    - 关节驱动的最大值 info_tuple[10]
                    - 关节的最大速度 info_tuple[11]
                    - 节点名称 info_tuple[12]
                    - 局部框架中的关节轴系 info_tuple[13]
                    - 父节点frame的关节位置 info_tuple[14]
                    - 父节点frame的关节方向 info_tuple[15]
                    - 父节点的索引，若是基座返回-1 info_tuple[16]
                
                - joint_num : 机器人的节点数量
        """
        joint_num = self.p.getNumJoints(self.robot_id)
        print("机器人的节点数量为：", joint_num)

        print("机器人的信息：")
        info_tuple_list = []
        for joint_index in range(joint_num):
            info_tuple = self.p.getJointInfo(self.robot_id, joint_index)
            info_tuple_list.append(info_tuple)
            print(f"组件{joint_index}内容\n {info_tuple}:")

            print(f"关节序号：{info_tuple[0]}\n\
                    关节名称：{info_tuple[1]}\n\
                    关节类型：{info_tuple[2]}\n\
                    机器人第一个位置的变量索引：{info_tuple[3]}\n\
                    机器人第一个速度的变量索引：{info_tuple[4]}\n\
                    保留参数：{info_tuple[5]}\n\
                    关节的阻尼大小：{info_tuple[6]}\n\
                    关节的摩擦系数：{info_tuple[7]}\n\
                    slider和revolute(hinge)类型的位移最小值：{info_tuple[8]}\n\
                    slider和revolute(hinge)类型的位移最大值：{info_tuple[9]}\n\
                    关节驱动的最大值：{info_tuple[10]}\n\
                    关节的最大速度：{info_tuple[11]}\n\
                    节点名称：{info_tuple[12]}\n\
                    局部框架中的关节轴系：{info_tuple[13]}\n\
                    父节点frame的关节位置：{info_tuple[14]}\n\
                    父节点frame的关节方向：{info_tuple[15]}\n\
                    父节点的索引，若是基座返回-1：{info_tuple[16]}\n\n")

        return {"info_tuple_list": info_tuple_list, "joint_num": joint_num}

    def set_camera(self, cameraDistance, cameraYaw, cameraPitch):
        """
            控制相机

            Parameters : 
                - cameraDistance : 相机的距离
                - cameraYaw : 相机绕垂直轴旋转的角度
                - cameraPitch : 相机绕水平轴旋转的角度
            
            example:
                3,110,-30

        """
        location, _ = self.p.getBasePositionAndOrientation(self.robot_id)
        self.p.resetDebugVisualizerCamera(cameraDistance=cameraDistance,
                                          cameraYaw=cameraYaw,
                                          cameraPitch=cameraPitch,
                                          cameraTargetPosition=location)

    def set_Joint(self,
                  target_btn,
                  max_force_btn,
                  available_joints_indexes,
                  joint_name="wheel"):
        """
            控制关节

            Parameters : 
                - target_v : 电机达到的预定角速度（rad/s）
                - max_force : 电机能够提供的力，这个值决定了机器人运动时的加速度，太快会翻车哟，单位N
                - available_joints_indexes : 机器人可用关节索引
            
            example:
                target_v=10,max_force=10

        """

        target_v = self.p.readUserDebugParameter(target_btn)
        max_force = self.p.readUserDebugParameter(max_force_btn)
        wheel_joints_indexes = [
            i for i in available_joints_indexes
            if joint_name in str(self.p.getJointInfo(self.robot_id, i)[1])
        ]

        # 控制多个关节
        self.p.setJointMotorControlArray(
            bodyUniqueId=self.robot_id,
            jointIndices=wheel_joints_indexes,
            controlMode=self.p.VELOCITY_CONTROL,
            targetVelocities=[target_v for _ in wheel_joints_indexes],
            forces=[max_force for _ in wheel_joints_indexes])

        #setJointMotorControl2 #控制单个关节

    def add_btn(self, paramName, rangeMin=1, rangeMax=0, startValue=0):
        """
            增加按钮,如果最大值为0，则为按钮，否则为滑块

            Parameters : 
                - paramName : 按钮名称
                - rangeMin : 按钮的最小值
                - rangeMax : 按钮的最大值
                - startValue : 按钮的初始值

            Returns:
                - btn : 按钮配置
                - btn_id : 返回值是滑块中参数的最新 按钮的读取值 。对于一个按钮，每按下一个按钮，按钮的参数就会增加1。
        """
        btn = self.p.addUserDebugParameter(paramName=paramName,
                                           rangeMin=rangeMin,
                                           rangeMax=rangeMax,
                                           startValue=startValue)
        btn_id = self.p.readUserDebugParameter(btn)
        return btn, btn_id

    def reset_init(self, btn, previous_btn):
        """
            获取按钮值，如果变化，重置模型位置

            Returns:
                - previous_btn : 最新读取值
        """
        if self.p.readUserDebugParameter(btn) != previous_btn:
            # 重置速度
            # for i in range(p.getNumJoints(self.robot_id)):
            #     p.setJointMotorControl2(self.robot_id, i, p.VELOCITY_CONTROL, 0, 0)

            # 重置位置
            # self.p.resetBasePositionAndOrientation(self.robot_id, self.startPos, self.startOrientation)

            # 移除当前加载的模型
            # self.p.removeBody(self.robot_id)
            # if self.object_id:
            #     self.p.removeBody(self.object_id)

            # 初始化环境
            self.p.resetSimulation(self.physicsClientId)
            self.init_config()

            previous_btn = self.p.readUserDebugParameter(btn)
        return previous_btn
