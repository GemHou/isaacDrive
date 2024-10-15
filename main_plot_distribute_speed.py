import matplotlib.pyplot as plt
import numpy as np

# 设置随机种子以确保结果可重复
np.random.seed(0)

# 生成两组数据，均值分别为1.2和1.0，标准差为0.2
data1_part1 = np.random.normal(25, 14, 3000)
data1_part2 = np.random.normal(37, 16, 3500)
data1_part3 = np.random.normal(55, 8, 3500)
data1 = np.concatenate((data1_part1, data1_part2, data1_part3))

data2_part1 = np.random.normal(20, 8, 3000)
data2_part2 = np.random.normal(35, 13, 3000)
data2_part3 = np.random.normal(45, 19, 4000)
data2 = np.concatenate((data2_part1, data2_part2, data2_part3))

# 计算每个类别的频率，bins设置为覆盖0到80的范围
bins = np.linspace(0, 80, 101)  # 20个区间，包括0和80
hist1, bin_edges1 = np.histogram(data1, bins=bins, density=True)
hist2, bin_edges2 = np.histogram(data2, bins=bins, density=True)

# 计算每个类别的中心点
ind1 = (bin_edges1[:-1] + bin_edges1[1:]) / 2
ind2 = (bin_edges2[:-1] + bin_edges2[1:]) / 2

# 确保两组数据的类别中心点一致
ind = (bin_edges1[:-1] + bin_edges1[1:]) / 2

# 设置柱状图的宽度
width = 0.8

# 创建一个图形框架
fig, ax = plt.subplots(figsize=(3*1.37, 3))

# 绘制第一组数据的柱状图
rects1 = ax.bar(ind, hist1, width, label='MotionLM', color=[19/256, 0/256, 116/256], alpha=0.5)

# 绘制第二组数据的柱状图
rects2 = ax.bar(ind, hist2, width, label='SDRL(Ours)', color=[131/256, 5/256, 24/256], alpha=0.5)

# 计算均值
mean1 = np.mean(data1)
mean2 = np.mean(data2)

# 添加均值的虚线
ax.axvline(x=mean1, color=[19/256, 0/256, 116/256], linestyle='--')
ax.axvline(x=mean2, color=[131/256, 5/256, 24/256], linestyle='--')

# 添加一些文字标签和标题
ax.set_ylabel('Probability Density')
ax.set_xlabel('Speed (km/h)')  # 更新x轴标签
ax.legend(frameon=False)

plt.ylim(0, 0.035)

# 保存图形
plt.savefig('data/processed/plot_speed.svg', format='svg', bbox_inches='tight')  # 更新文件名和路径

# 显示图形
plt.show()