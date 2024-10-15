import matplotlib.pyplot as plt
import numpy as np

# 设置随机种子以确保结果可重复
np.random.seed(0)

# 生成两组数据，均值分别为1.2和1.0，标准差为0.2
data1 = np.random.normal(1.2, 0.2, 1000)
data2 = np.random.normal(1.0, 0.2, 1000)

# 计算每个类别的频率
hist1, bin_edges1 = np.histogram(data1, bins=5, density=True)
hist2, bin_edges2 = np.histogram(data2, bins=5, density=True)

# 计算每个类别的中心点
ind1 = (bin_edges1[:-1] + bin_edges1[1:]) / 2
ind2 = (bin_edges2[:-1] + bin_edges2[1:]) / 2

# 确保两组数据的类别中心点一致
ind = ind1

# 设置柱状图的宽度
width = 0.4

# 创建一个图形框架
fig, ax = plt.subplots(figsize=(3*1.37, 3))

# 绘制第一组数据的柱状图
rects1 = ax.bar(ind - width / 2, hist1, width, label='MotionLM', color=[19/256, 0/256, 116/256])

# 绘制第二组数据的柱状图，位置稍微偏移
rects2 = ax.bar(ind + width / 2, hist2, width, label='SDRL(Ours)', color=[131/256, 5/256, 24/256])

# 添加一些文字标签和标题
ax.set_ylabel(r'Probability Density')
ax.set_xlabel('Time to Collision')
ax.set_xticks(ind)
ax.set_xticklabels(('Vehicle\nCollision', 'Pedestrian\nCollision',
                    'Road\nCollision',
                    'Extremely\nBrake', 'Unstable\nWheel'), fontsize=8.5)
ax.legend(frameon=False)

# 保存图形
plt.savefig('data/processed/plot_ttc.svg', format='svg', bbox_inches='tight')

# 显示图形
plt.show()