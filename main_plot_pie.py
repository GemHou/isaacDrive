import matplotlib.pyplot as plt
import numpy as np

# 这是你提供的数据
data = [1433, 433, 1075, 2297, 970, 434, 2418, 457]

# 创建一个标签列表，用于饼图的每个部分
labels = ['Cut-in', 'Brake', 'Lane-change', 'Cross', 'Merge', 'Start-up', 'Cruise', 'U-turn']
labels2 = ['Cut-in\n1433', 'Brake\n433', 'Lane-change\n1075', 'Cross\n2297', 'Combine\n970', 'Start-up\n434',
           'Cruise\n2418', 'U-turn\n457']

# 定义两种颜色
color1 = [19 / 256 + 0.15, 0 / 256 + 0.15, 116 / 256 + 0.15]
color2 = [131 / 256 + 0.15, 5 / 256 + 0.15, 24 / 256 + 0.15]

# 创建颜色列表，交替使用两种颜色，并添加随机变化
colors = []
for i in range(len(data)):
    if i % 2 == 0:
        base_color = color1
    else:
        base_color = color2
    # 添加一些随机变化，范围在-0.1到0.1之间
    random_color = [max(0, min(1, base_color[j] + np.random.uniform(-0.05, 0.05))) for j in range(3)]
    colors.append(random_color)

# 绘制饼图
plt.figure(figsize=(3 * 1.37, 3))
plt.pie(data,
        labels=labels2,
        autopct=lambda pct: f"{pct:.0f}%",
        startangle=140,
        colors=colors,
        pctdistance=0.8,
        labeldistance=1.15
        )

# # 在每个部分旁边添加原始数值
# for i, value in enumerate(data):
#     plt.text(i, 0, f"{value}", ha='center', va='center', color='white', size=10)

# 保存图形
plt.savefig('data/processed/plot_pie.svg', format='svg', bbox_inches='tight')

# 显示图形
plt.show()
