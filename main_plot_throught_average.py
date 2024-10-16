import matplotlib.pyplot as plt
import numpy as np

# 假设这是你的两组数据
data1 = np.array([1948, 1572, 1433, 1221, 976, 801, 591, 349, 289])
data2 = np.array([1027, 944, 961, 937, 937, 884, 829, 749, 732])
data1_std = np.array([20.0, 17.0, 16.8, 17.0, 17.3, 23.5, 29.1, 12.9, 14.3])
data2_std = np.array([5.0, 68.0, 43.8, 32.1, 22.3, 48.0, 15.6, 36.4, 8.9])

# 创建一个索引数组，用于x轴的标签
ind = np.arange(len(data1))  # 这个长度应该和数据的长度一致

# 创建一个图形框架
fig, ax = plt.subplots(figsize=(3*1.37, 3))

# 绘制第一组数据的折线图
ax.plot(ind, data1, label='CPU Parallelism', color=[19/256, 0/256, 116/256])

# 绘制第二组数据的折线图
ax.plot(ind, data2, label='GPU Parallelism', color=[131/256, 5/256, 24/256])

# 根据std标准差绘制折线阴影
ax.fill_between(ind, data1 - data1_std * 3, data1 + data1_std * 3, color=[19/256, 0/256, 116/256], alpha=0.2)
ax.fill_between(ind, data2 - data2_std * 3, data2 + data2_std * 3, color=[131/256, 5/256, 24/256], alpha=0.2)

# 添加一些文字标签和标题
ax.set_ylabel(r'Throughput')  # 使用LaTeX语法设置上标
ax.set_xlabel('Batch size')
ax.set_xticks(ind)
ax.set_xticklabels(('1', '2', '4', '8', '16', '32', '64', '128', '160'))
ax.legend(frameon=False)

# 保存图形
plt.savefig('data/processed/plot_throughout_average.svg', format='svg', bbox_inches='tight')

# 显示图形
plt.show()