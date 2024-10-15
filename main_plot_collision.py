import matplotlib.pyplot as plt
import numpy as np

# 新的两组数据，百分比转换为小数形式
data1 = np.array([18.60, 17.90, 19.20, 18.50, 17.20, 16.50, 16.40, 16.10, 16.20, 16.50])
data2 = np.array([18.60, 14.10, 13.50, 12.30, 11.60, 10.80, 9.80, 10.20, 10.30, 10.10])

# 假设的标准差，这里需要您提供具体的数值，如果没有提供，我将使用假设值
data1_std = np.array([0.49, 0.65, 0.65, 1.01, 1.01, 0.44, 0.21*2, 0.15*2, 0.21*2, 0.21*2])
data2_std = np.array([0.49, 0.65, 0.65, 1.01, 1.01, 0.90, 0.50, 0.26*2, 0.10*4, 0.14*4])

# 创建一个索引数组，用于x轴的标签
ind = np.arange(len(data1))  # 这个长度应该和数据的长度一致

# 创建一个图形框架
fig, ax = plt.subplots(figsize=(3*1.37, 3))

# 绘制第一组数据的折线图
ax.plot(ind, data1, label='MotionLM', color=[19/256, 0/256, 116/256])

# 绘制第二组数据的折线图
ax.plot(ind, data2, label='SDRL(Ours)', color=[131/256, 5/256, 24/256])

# 根据std标准差绘制折线阴影
ax.fill_between(ind, data1 - data1_std * 1, data1 + data1_std * 1, color=[19/256, 0/256, 116/256], alpha=0.2)
ax.fill_between(ind, data2 - data2_std * 1, data2 + data2_std * 1, color=[131/256, 5/256, 24/256], alpha=0.2)

# 添加一些文字标签和标题
ax.set_ylabel('Collision rate (%)')  # 使用LaTeX语法设置上标
ax.set_xlabel('Training epochs')
ax.set_xticks(ind)
ax.set_xticklabels(('1', '2', '3', '4', '5', '6', '7', '8', '9', '10'))  # 更新x轴标签以匹配数据长度
ax.legend(frameon=False)

# 保存图形
plt.savefig('data/processed/plot_throughout_collision.svg', format='svg', bbox_inches='tight')

# 显示图形
plt.show()