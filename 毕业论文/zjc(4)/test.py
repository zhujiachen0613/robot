# 定义颜色范围
import numpy as np


color_ranges = {
    'yellow': ([30, 150, 150], [45, 255, 255]),
    'brown': ([10, 100, 100], [20, 255, 255]),
    'blue': ([110, 50, 50], [130, 255, 255]),
    'green': ([50, 50, 50], [80, 255, 255])
}

# 更新square_colors字典以确保颜色在给定的范围内
square_colors = {}

for color_name, (lower_bound, upper_bound) in color_ranges.items():
    # 生成颜色值
    color = tuple(np.random.randint(lower_bound, upper_bound))
    square_colors[color_name] = color

print(square_colors)