import matplotlib.pyplot as plt
import numpy as np

# 这是你提供的数据
data = [457, 1433, 433, 1075, 2297, 970, 434, 2418]

# 创建一个标签列表，用于饼图的每个部分
labels = ['Category 1', 'Category 2', 'Category 3', 'Category 4', 'Category 5', 'Category 6', 'Category 7', 'Category 8']

# 定义两种颜色
color1 = [19/256, 0/256, 116/256]
color2 = [131/256, 5/256, 24/256]

# 创建颜色列表，交替使用两种颜色，并添加随机变化
colors = []
for i in range(len(data)):
    if i % 2 == 0:
        base_color = color1
    else:
        base_color = color2
    # 添加一些随机变化，范围在-0.1到0.1之间
    random_color = [max(0, min(1, base_color[j] + np.random.uniform(-0.1, 0.1))) for j in range(3)]
    colors.append(random_color)

# 绘制饼图
plt.figure(figsize=(3*1.37, 3))
plt.pie(data, labels=labels, autopct='%1.1f%%', startangle=140, colors=colors)

# 添加标题
# plt.title('Pie Chart Example')

# 保存图形
plt.savefig('data/processed/plot_pie.svg', format='svg', bbox_inches='tight')

# 显示图形
plt.show()