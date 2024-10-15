import matplotlib.pyplot as plt

# 这是你提供的数据
data = [457, 1433, 433, 1075, 2297, 970, 434, 2418]

# 创建一个标签列表，用于饼图的每个部分
labels = ['Category 1', 'Category 2', 'Category 3', 'Category 4', 'Category 5', 'Category 6', 'Category 7', 'Category 8']

# 绘制饼图
plt.figure(figsize=(8, 8))
plt.pie(data, labels=labels, autopct='%1.1f%%', startangle=140)

# 添加标题
plt.title('Pie Chart Example')

plt.savefig('data/processed/plot_pie.svg', format='svg', bbox_inches='tight')

# 显示图形
plt.show()