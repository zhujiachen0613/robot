import serial
import time
import cv2
import numpy as np

# 定义关节气囊的色彩范围【色相，饱和度，明度】
lower_yellow = np.array([20, 150, 150])
upper_yellow = np.array([30, 255, 255])

lower_brown = np.array([10, 50, 50])
upper_brown = np.array([20, 255, 255])

lower_blue = np.array([100, 50, 50])
upper_blue = np.array([110, 255, 255])

lower_green = np.array([80, 50, 50])
upper_green = np.array([90, 255, 255])
# 配置串口
serial_com4 = "COM4"  # 串口
baud_rate = 9600  # 波特率
ser = serial.Serial(serial_com4, baud_rate, timeout=0.5)
# 检查串口是否连接成功
if ser.is_open:
    print("已连接到串口")
else:
    print("未连接到串口")

# 配置摄像头
cap = cv2.VideoCapture(0)
if cap.isOpened():
    ref, frame = cap.read()
    print("摄像头已启动")
else:
    print("摄像头未启动")
    ref = False


# 按照颜色识别轮廓
def color_counter(lower, upper, img, hsv):  # 返回轮廓矩形
    mask = cv2.inRange(hsv, lower, upper)
    frame_masked = cv2.bitwise_and(img, img, mask=mask)
    gray = cv2.cvtColor(frame_masked, cv2.COLOR_BGR2GRAY)
    _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    contours, _ = cv2.findContours(thresh, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    # 计算每个轮廓的面积
    contour_areas = [cv2.contourArea(contour) for contour in contours]
    # 找到最大面积的轮廓索引
    max_area_index = contour_areas.index(max(contour_areas))
    # 获取最大面积的轮廓
    max_area_contour = contours[max_area_index]
    # 找到最小外接矩形
    rect = cv2.minAreaRect(max_area_contour)
    # 获取矩形四个角的坐标
    pts = cv2.boxPoints(rect)
    pts = np.intp(pts)  # 转换为整数，以便后续处理
    return pts


# 按照矩形四个点绘制矩形轮廓
def draw_couter(pts, img):
    for i in range(4):
        cv2.line(img, pts[i], pts[(i + 1) % 4], (0, 0, 255), 2)  # 用红色线条画矩形框
        cv2.line(img, pts[i], pts[(i + 2) % 4], (0, 255, 0), 2)  # 用绿色线条画矩形框


# 识别四个气囊的轮廓
def capture_counter(img):
    # 将图像转换为HSV格式，以更好地检测颜色
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    # pts = np.int0(pts)  # 转换为整数，以便后续处理
    pts_yellow = color_counter(lower_yellow, upper_yellow, img, hsv)
    pts_blue = color_counter(lower_blue, upper_blue, img, hsv)
    pts_brown = color_counter(lower_brown, upper_brown, img, hsv)
    pts_green = color_counter(lower_green, upper_green, img, hsv)
    # 用矩形框住轮廓（假设你的图像是 img）
    draw_couter(pts_yellow, img)
    draw_couter(pts_blue, img)
    draw_couter(pts_brown, img)
    draw_couter(pts_green, img)


# 发送数据给串口
def apply_action(action, value):
    ser.write(str(action).encode())
    time.sleep(0.1)
    ser.write(str(value).encode())
    time.sleep(5)


# 时刻检查串口数据输入
def load_com4_data():
    while True:
        if ser.in_waiting > 0:
            line = ser.readline().decode("utf-8").rstrip()
            print(line)


print("hello world")

# 从摄像头读取一张照片
# ret, frame = cap.read()
# if not ret:
#     print("无法获取图像")
#     exit()
# # 将图像保存为jpg格式
# cv2.imwrite('capture.jpg', frame)

# 读取刚保存的图像
img = cv2.imread("capture.jpg")
capture_counter(img)
cv2.imwrite("capture1.jpg", img)
# cv2.imshow('Image with contours', img)
# time.sleep(10)
# # 释放摄像头资源并关闭窗口
# cv2.destroyAllWindows()
