import numpy as np
import matplotlib.pyplot as plt

# 准备数据
labels = np.array(['Safety', 'Panic', 'Comfort', 'Speed', 'Energy', 'Rule'])
data1 = np.array([0.8, 0.7, 0.8, 0.9, 1.0, 1.0])
data2 = np.array([1.4, 1.5, 1.2, 1.1, 0.95, 0.95])

# 设置雷达图的角度
angles = np.linspace(0, 2 * np.pi, len(labels), endpoint=False).tolist()

# 使图形闭合
data1 = np.concatenate((data1, [data1[0]]))
data2 = np.concatenate((data2, [data2[0]]))
angles += angles[:1]

# 绘制雷达图
fig, ax = plt.subplots(figsize=(3*1.37, 3), subplot_kw=dict(polar=True))
ax.fill(angles, data1, color='red', alpha=0.25)
ax.fill(angles, data2, color='blue', alpha=0.25)
ax.plot(angles, data1, color='red', linewidth=2, linestyle='solid', label='MotionLM')
ax.plot(angles, data2, color='blue', linewidth=2, linestyle='solid', label='SDRL(Ours)')

# 添加标签
ax.set_xticks(angles[:-1])
ax.set_xticklabels(labels)

# 添加图例
plt.legend(loc='upper right', bbox_to_anchor=(0.1, 0.1), frameon=False)

plt.savefig('data/processed/plot_radar.svg', format='svg', bbox_inches='tight')

# 显示图表
plt.show()