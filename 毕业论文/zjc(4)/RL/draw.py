import pandas as pd
import matplotlib.pyplot as plt

# 读取CSV文件
df = pd.read_csv('../jieguo/第一次不给负奖励/danci_reward_50000.csv')

# 假设CSV文件的四列分别是：'A', 'B', 'C', 'D'
# 获取列名
columns = df.columns.tolist()

# 绘制折线图
plt.figure(figsize=(10, 6))  # 设置图表大小
for column in columns:
    plt.plot(df[column], label=column,linewidth=1)  # 为每列数据绘制折线图，并添加图例

plt.title('Action Data')  # 设置图表标题
plt.xlabel('Index')  # 设置x轴标签
plt.ylabel('Value')  # 设置y轴标签
plt.legend()  # 显示图例
plt.grid(True)  # 显示网格
plt.show()  # 显示图表