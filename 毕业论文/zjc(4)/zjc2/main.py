"""
    运行pybullet主文件
"""
import argparse
import time
from utils.mybullet import PyBulletModel


def print_info(bullet):
    while True:
        user_input = input("1. 打印机器人位置\n"
                           "2. 打印机器人信息\n"
                           "3. 获取使用的关节\n"
                           "输入（输入q退出）：")
        options = {
            "1": bullet.get_robot_location,
            "2": bullet.get_robot_info,
            "3": bullet.get_use_joints,
        }

        if user_input.lower() == "q":
            break

        action = options.get(user_input)
        if action:
            action()
        else:
            print("无效的输入")
        # 添加自定义按钮的回调函数


if __name__ == '__main__':
    # parser = argparse.ArgumentParser(description="PyBullet模型运行程序")
    # parser.add_argument("modelpath",type=str,help="模型路径")
    # parser.add_argument("--gui", action="store_true", help="开启界面渲染")
    # args = parser.parse_args()
    # python main.py "/home/hp/zjc/robot/urdf/robot.urdf" --gui

    modelpath = "/home/hp/zjc/robot/urdf/robot.urdf"
    objectpath = "/home/hp/zjc/pp/urdf/pp.urdf"
    gui_input = input("请选择是否开启界面渲染，y/n ：")
    # bullet = PyBulletModel(modelpath, gui_input, [0, 0, 0], [0, 0, 0],
    #                        objectpath)
    bullet = PyBulletModel("r2d2.urdf", gui_input, [0, 0, 1], [0, 0, 0])

    # 获取控制关节
    joints = bullet.get_use_joints()
    # 创建重置按钮
    btn, btn_reset_object = bullet.add_btn("reset")

    # 读取参数的值，读取速度，转向角度，驱动力参数
    targetVelocity_btn, targetVelocitySlider = bullet.add_btn(
        "wheelVelocity", -30, 50, 0)
    maxForce_btn, maxForceSlider = bullet.add_btn("maxForce", 0, 100, 10)
    # steering_btn, steeringSlider = bullet.add_btn("steering", -0.5, 0.5, 0)

    text_id = bullet.add_text(text="machine project",
                              textPosition=[0, 1, 3],
                              textColorRGB=[0, 1, 0],
                              textSize=1.2)  # 增加文字
    # print_info(bullet)

    for i in range(100000000):
        bullet.stepSimulation()
        bullet.set_camera(3, 110, -30)
        bullet.set_Joint(targetVelocity_btn, maxForce_btn,
                         joints["available_joints_indexes"], "wheel")
        btn_reset_object = bullet.reset_init(btn, btn_reset_object)
        time.sleep(1 / 240)

    bullet.p.disconnect()
