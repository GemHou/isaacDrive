import matplotlib.pyplot as plt
import numpy as np

# 假设这是你的两组数据
data1 = [1948, 3144, 5735, 9775, 15618, 25654, 37850, 44685, 46356]
data2 = [1027, 1888, 3845, 7501, 15006, 28304, 53108, 95927, 117172]

# 创建一个索引数组，用于x轴的标签
ind = np.arange(len(data1))  # 这个长度应该和数据的长度一致

# 设置柱状图的宽度
width = 0.35

# 创建一个图形框架
fig, ax = plt.subplots()

# 绘制第一组数据的柱状图
rects1 = ax.bar(ind, data1, width, label='Category 1')

# 绘制第二组数据的柱状图，位置稍微偏移
rects2 = ax.bar(ind + width, data2, width, label='Category 2')

# 添加一些文字标签和标题
ax.set_ylabel('Scores')
ax.set_title('Scores by category')
ax.set_xticks(ind + width / 2)
ax.set_xticklabels(('1', '2', '4', '8', '16', '32', '64', '128', '160'))
ax.legend()

# 在柱状图上方添加数值标签
def autolabel(rects):
    """在每个柱状图上方添加数值标签"""
    for rect in rects:
        height = rect.get_height()
        ax.annotate('{}'.format(height),
                    xy=(rect.get_x() + rect.get_width() / 2, height),
                    xytext=(0, 3),  # 3点垂直偏移
                    textcoords="offset points",
                    ha='center', va='bottom')

autolabel(rects1)
autolabel(rects2)

# 显示图形
plt.show()
