import matplotlib.pyplot as plt
import numpy as np

# 假设这是你的两组数据
data1 = np.array([-7985, -5371, -2795, -1865, -1415, -445, -321, 255, 523, 737, 995, 1168, 1316, 1456, 1611, 1694, 1726, 1765, 1823, 1861, 1904])

data2 = np.array([-10080, -9963, - 9792, -9608, - 9090, -8957, - 8649, -8978, - 8233, -7244, - 6717, -6703, - 6060, -5492, - 4706, -4470, - 3880, -3365, - 3298, -3092, - 2547])
data1_std = np.array([144.8,	205.9,	388.1,	439.5,	471.7,	349.1,	347.5,	520.2,	504.1,	364.2,	320.6,	280.6,	240.5,	213.9,	172.1,	122.6,	79.3,	68.5,	71.6,	59.0,	40.5])
data2_std = np.array([144.8,	205.9,	388.1,	439.5,	471.7,	349.1,	347.5,	719.3,	954.4,	1000.9,	811.1,	677.4,	854.9,	928.9,	859.7,	811.0,	635.2,	556.2,	482.4,	371.0,	388.0])

# 创建一个索引数组，用于x轴的标签
ind = np.arange(0, len(data1)*5, 1*5)  # 这个长度应该和数据的长度一致

# 创建一个图形框架
fig, ax = plt.subplots(figsize=(3*1.37, 3))

# 绘制第一组数据的折线图
ax.plot(ind, data1, label='w. Gradient', color=[19/256, 0/256, 116/256])

# 绘制第二组数据的折线图
ax.plot(ind, data2, label='w/o. Gradient', color=[131/256, 5/256, 24/256])

# 根据std标准差绘制折线阴影
ax.fill_between(ind, data1 - data1_std * 3, data1 + data1_std * 3, color=[19/256, 0/256, 116/256], alpha=0.2)
ax.fill_between(ind, data2 - data2_std * 3, data2 + data2_std * 3, color=[131/256, 5/256, 24/256], alpha=0.2)

# 添加一些文字标签和标题
ax.set_ylabel(r'Reward')  # 使用LaTeX语法设置上标
ax.set_xlabel('Training iterations')
# ax.set_xticks(np.arange(0, 21, 1))  # 设置x轴刻度为0到100，步长为10
# ax.set_xticklabels((f'{i}' for i in range(0, 101, 10)))  # 设置x轴刻度标签
ax.legend(frameon=False)

# 保存图形
plt.savefig('data/processed/plot_grad.svg', format='svg', bbox_inches='tight')

# 显示图形
plt.show()