import matplotlib.pyplot as plt
import numpy as np

# 假设这是你的两组数据
data1 = np.array([11.6, 9.3, 11.3, 7.3])
data2 = np.array([12.7, 9.5, 10.9, 6.9])
data3 = np.array([10.6, 8.5, 10.8, 6.5])

# 创建一个索引数组，用于x轴的标签
ind = np.arange(len(data1))  # 这个长度应该和数据的长度一致

# 设置柱状图的宽度
width = 0.3

# 创建一个图形框架
fig, ax = plt.subplots(figsize=(3*1.37, 3))

# 绘制第一组数据的柱状图
rects1 = ax.bar(ind, data1, width, label='SDRL w/o. RewardGradient', color=[19/256, 0/256, 116/256])

# 绘制第二组数据的柱状图，位置稍微偏移
rects2 = ax.bar(ind + width, data2, width, label='SDRL w/o. InteractiveEnvironment', color=[75/256, 2/256, 70/256])

rects3 = ax.bar(ind + width * 2, data3, width, label='SDRL (Ours)', color=[131/256, 5/256, 24/256])

# 添加一些文字标签和标题
ax.set_ylabel(r'Disengagement Rate $(\%)$')  # 使用LaTeX语法设置上标
# ax.set_xlabel('Batch size')
ax.set_xticks(ind + width / 2)
ax.set_xticklabels(('Cut-in', 'Cross',
                    'Lane-change',
                    'Cruise'), fontsize=8.5)
ax.legend(frameon=False)
ax.set_ylim(0, 19)

plt.savefig('data/processed/plot_ablation.svg', format='svg', bbox_inches='tight')

# 显示图形
plt.show()