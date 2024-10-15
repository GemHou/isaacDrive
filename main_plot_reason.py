import matplotlib.pyplot as plt
import numpy as np

# 假设这是你的两组数据
data1 = np.array([74.9/1273, 179.6/1273, 648/1273, 636.5/1273, 648/1273]) * 10
data2 = np.array([33.1/1273, 141.4/1273, 628.5/1273, 424.3/1273, 281/1273]) * 10

# 创建一个索引数组，用于x轴的标签
ind = np.arange(len(data1))  # 这个长度应该和数据的长度一致

# 设置柱状图的宽度
width = 0.4

# 创建一个图形框架
fig, ax = plt.subplots(figsize=(3*1.37, 3))

# 绘制第一组数据的柱状图
rects1 = ax.bar(ind, data1, width, label='MotionLM', color=[19/256, 0/256, 116/256])

# 绘制第二组数据的柱状图，位置稍微偏移
rects2 = ax.bar(ind + width, data2, width, label='SDRL(Ours)', color=[131/256, 5/256, 24/256])

# 添加一些文字标签和标题
ax.set_ylabel(r'Disengagement Rate $( \times 10^{-4})$')  # 使用LaTeX语法设置上标
# ax.set_xlabel('Batch size')
ax.set_xticks(ind + width / 2)
ax.set_xticklabels(('Vehicle\nCollision', 'Pedestrian\nCollision',
                    'Road\nCollision',
                    'Extremely\nBrake', 'Unstable\nWheel'), fontsize=8)
ax.legend(frameon=False)

plt.savefig('data/processed/plot_reason.svg', format='svg', bbox_inches='tight')

# 显示图形
plt.show()