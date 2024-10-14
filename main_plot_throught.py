import matplotlib.pyplot as plt
import numpy as np

# 假设这是你的两组数据
data1 = [1948, 3144, 5735, 9775, 15618, 25654, 37850, 44685, 46356]
data2 = [1027, 1888, 3845, 7501, 15006, 28304, 53108, 95927, 117172]

data1 = [x/1000 for x in data1]
data2 = [x/1000 for x in data2]

# 创建一个索引数组，用于x轴的标签
ind = np.arange(len(data1))  # 这个长度应该和数据的长度一致

# 设置柱状图的宽度
width = 0.4

# 创建一个图形框架
fig, ax = plt.subplots(figsize=(3*1.37, 3))

# 绘制第一组数据的柱状图
rects1 = ax.bar(ind, data1, width, label='CPU Parallelism', color=[19/256, 0/256, 116/256])

# 绘制第二组数据的柱状图，位置稍微偏移
rects2 = ax.bar(ind + width, data2, width, label='GPU Parallelism', color=[131/256, 5/256, 24/256])

# 添加一些文字标签和标题
ax.set_ylabel(r'Throughput $( \times 10^3 )$')  # 使用LaTeX语法设置上标
ax.set_xlabel('Batch size')
# ax.set_title('Scores by category (log scale)')
ax.set_xticks(ind + width / 2)
ax.set_xticklabels(('1', '2', '4', '8', '16', '32', '64', '128', '160'))
ax.legend(frameon=False)

# 设置纵坐标为对数坐标
# ax.set_yscale('log')

# # 在柱状图上方添加数值标签
# def autolabel(rects):
#     """在每个柱状图上方添加数值标签"""
#     for rect in rects:
#         height = rect.get_height()
#         ax.annotate('{}'.format(height),
#                     xy=(rect.get_x() + rect.get_width() / 2, height),
#                     xytext=(0, 3),  # 3点垂直偏移
#                     textcoords="offset points",
#                     ha='center', va='bottom', fontsize=5.5)

# autolabel(rects1)
# autolabel(rects2)

plt.savefig('data/processed/plot_throughout.pdf', format='pdf', bbox_inches='tight')
plt.savefig('data/processed/plot_throughout.svg', format='svg', bbox_inches='tight')
plt.savefig('data/processed/plot_throughout.eps', format='eps', bbox_inches='tight')
plt.savefig('data/processed/plot_throughout.tif', format='tif', bbox_inches='tight')

# 显示图形
plt.show()