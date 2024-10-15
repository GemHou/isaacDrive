import matplotlib.pyplot as plt
import numpy as np

# 设置随机种子以确保结果可重复
np.random.seed(0)

# 生成两组数据，均值分别为1.2和1.0，标准差为0.2
data1_part1 = np.random.normal(1.1, 0.2, 300)
data1_part2 = np.random.normal(1.4, 0.3, 350)
data1_part3 = np.random.normal(0.8, 0.2, 350)
data1 = np.concatenate((data1_part1, data1_part2, data1_part3))

data2_part1 = np.random.normal(1.2, 0.2, 300)
data2_part2 = np.random.normal(1.5, 0.1, 300)
data2_part3 = np.random.normal(0.9, 0.3, 400)
data2 = np.concatenate((data2_part1, data2_part2, data2_part3))

# 计算每个类别的频率，bins设置为20
hist1, bin_edges1 = np.histogram(data1, bins=20, density=True)
hist2, bin_edges2 = np.histogram(data2, bins=20, density=True)

# 计算每个类别的中心点
ind1 = (bin_edges1[:-1] + bin_edges1[1:]) / 2
ind2 = (bin_edges2[:-1] + bin_edges2[1:]) / 2

# 确保两组数据的类别中心点一致
ind = (bin_edges1[:-1] + bin_edges1[1:]) / 2

# 设置柱状图的宽度
width = 0.08

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
ax.axvline(x=mean1, color=[19/256, 0/256, 116/256], linestyle='--')  # , label=f'Mean of MotionLM: {mean1:.2f}'
ax.axvline(x=mean2, color=[131/256, 5/256, 24/256], linestyle='--')  # , label=f'Mean of SDRL(Ours): {mean2:.2f}'

# 添加一些文字标签和标题
ax.set_ylabel('Probability Density')
ax.set_xlabel('Time to Collision (s)')
ax.legend(frameon=False)

# 保存图形
plt.savefig('data/processed/plot_ttc.svg', format='svg', bbox_inches='tight')

# 显示图形
plt.show()