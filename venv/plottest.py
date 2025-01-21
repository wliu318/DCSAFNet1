import matplotlib.pyplot as plt
import numpy as np

# 获取 'Paired' 颜色映射
cmap = plt.cm.get_cmap('Paired')

# 生成一个包含 'Paired' 颜色映射中所有颜色的数组
colors = cmap(np.linspace(0, 1, cmap.N))  # cmap.N 是颜色映射中的颜色数量

# 创建一个图形和坐标轴
fig, ax = plt.subplots()

# 显示颜色条
ax.imshow([colors], extent=[0, 10, 0, 1], aspect='auto')
ax.axis('off')  # 关闭坐标轴

# 显示图形  
plt.show()