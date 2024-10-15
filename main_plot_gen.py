import matplotlib.pyplot as plt
import numpy as np

# 原始数据
data1 = np.array([18.60, 14.10, 13.50, 12.30, 11.60, 10.80, 9.80, 10.20, 10.30, 10.10])

# 创建一个索引数组，用于x轴的标签
ind = np.arange(len(data1))  # 这个长度应该和数据的长度一致

# 创建一个图形框架
fig, ax = plt.subplots(figsize=(3*1.37, 3))

# 绘制原始数据的折线图
ax.plot(ind, data1, label='Vehicle SAIC LS6', color=[19/256, 0/256, 116/256], marker="o", alpha=0.5)

# 生成随机波动的数据
np.random.seed(0)  # 设置随机种子以确保结果可复现
data2 = data1 + np.random.normal(0, 0.5, len(data1)) + 0.1
data3 = data1 + np.random.normal(0, 0.5, len(data1)) - 0.1
data4 = data1 + np.random.normal(0, 0.5, len(data1)) + 0.5
data5 = data1 + np.random.normal(0, 0.5, len(data1)) - 0.5

# 绘制随机波动的数据
ax.plot(ind, data2, label='Vehicle SAIC LS6 + Noise 1', marker="o", color=[131/256+0.2, 5/256, 24/256], alpha=0.5)
ax.plot(ind, data3, label='Vehicle SAIC LS6 + Noise 2', marker="o", color=[19/256, 0/256+0.2, 116/256], alpha=0.5)
ax.plot(ind, data4, label='Vehicle SAIC LS6 + Noise 3', marker="o", color=[131/256, 5/256, 24/256+0.2], alpha=0.5)
ax.plot(ind, data5, label='Vehicle SAIC LS6 + Noise 4', marker="o", color=[19/256+0.2, 0/256+0.2, 116/256+0.2], alpha=0.5)

# 添加一些文字标签和标题
ax.set_ylabel('Collision rate (%)')  # 使用LaTeX语法设置上标
ax.set_xlabel('Training epochs')
ax.set_xticks(ind)
ax.set_xticklabels(('1', '2', '3', '4', '5', '6', '7', '8', '9', '10'))  # 更新x轴标签以匹配数据长度
ax.legend(frameon=False)

# 保存图形
plt.savefig('data/processed/plot_throughout_gen.svg', format='svg', bbox_inches='tight')

# 显示图形
plt.show()