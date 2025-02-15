import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

sns.set_palette("Set2")

# # 示例数据
# df1 = pd.DataFrame({
#     'x': [1, 2, 4, 5, 6, 7, 8, 9, 10],
#     'y1': [34.74, 52.07, 58.02, 57.51, 57.09, 56.98, 56.29, 55.90, 55.37],
#     'y2': [18.21, 44.41, 57.95, 57.82, 57.35, 57.05, 55.80, 55.09, 54.55],
#     'y3': [36.43, 44.80, 48.46, 49.04, 49.24, 49.69, 49.70, 49.32, 48.80]
# })

# df2 = pd.DataFrame({
#     'x': [1, 2, 4, 5, 6, 7, 8, 9, 10],
#     'y3': [4.79, 3.04, 2.58, 2.57, 2.59, 2.60, 2.63, 2.64, 2.65]
# })

# # 创建一个图形和第一个轴
# fig, ax1 = plt.subplots()

# # 在第一个轴上绘制前两个数据集
# sns.lineplot(data=df1, x='x', y='y1', ax=ax1, label='Avg Acc')
# sns.lineplot(data=df1, x='x', y='y2', ax=ax1, label='Avg UAcc')
# sns.lineplot(data=df1, x='x', y='y3', ax=ax1, label='Test Acc')
# ax1.set_ylabel('Accuracy')
# ax1.tick_params(axis='y')
# ax1.set_xlabel("Number of Models Merged")

# # 创建第二个轴，共享x轴
# ax2 = ax1.twinx()

# # 在第二个轴上绘制第三个数据集
# sns.lineplot(data=df2, x='x', y='y3', ax=ax2, label='Set Size', color='b')
# ax2.set_ylabel('Set Size')
# ax2.tick_params(axis='y')

# # 显示图例
# # fig.legend(loc='upper left', bbox_to_anchor=(0.1, 0.9))

# ax1.grid(True)
# ax2.grid(False)  # 防止双轴情况下重复显示网格线

# plt.savefig('uncertainty.png', dpi=300, bbox_inches='tight')

# # 示例数据
# df1 = pd.DataFrame({
#     'x': ["okvqa", "ocrvqa", "gqa", "vqav2", "textcaps", "a_okvqa", "refcoco", "llava_instruct", "sharegpt", "vg"],
#     'y1': [56.50, 44.52, 34.74, 46.15, 27.85, 56.50, 41.81, 41.08, 39.18, 39.46],
#     'y2': [57.26, 32.78, 18.21, 40.66, 15.70, 57.26, 34.40, 31.16, 31.56, 29.16],
#     'y3': [27.48, 35.21, 36.43, 35.29, 7.46, 20.26, 20.08, 21.17, 20.65, 26.63]
# })

# df2 = pd.DataFrame({
#     'x': ["okvqa", "ocrvqa", "gqa", "vqav2", "textcaps", "a_okvqa", "refcoco", "llava_instruct", "sharegpt", "vg"],
#     'y3': [2.54, 3.45, 4.79, 2.92, 4.40, 2.54, 3.13, 3.34, 3.18, 3.43]
# })

# # 创建一个图形和第一个轴
# fig, ax1 = plt.subplots()

# # 在第一个轴上绘制前两个数据集
# sns.lineplot(data=df1, x='x', y='y1', ax=ax1, label='Avg Acc')
# sns.lineplot(data=df1, x='x', y='y2', ax=ax1, label='Avg UAcc')
# sns.lineplot(data=df1, x='x', y='y3', ax=ax1, label='Test Acc')
# ax1.set_ylabel('Accuracy')
# ax1.tick_params(axis='y')
# ax1.set_xlabel("Number of Models Merged")

# # 创建第二个轴，共享x轴
# ax2 = ax1.twinx()

# # 在第二个轴上绘制第三个数据集
# sns.lineplot(data=df2, x='x', y='y3', ax=ax2, label='Set Size', color='b')
# ax2.set_ylabel('Set Size')
# ax2.tick_params(axis='y')

# # 显示图例
# # fig.legend(loc='upper left', bbox_to_anchor=(0.1, 0.9))

# ax1.grid(True)
# ax2.grid(False)  # 防止双轴情况下重复显示网格线

# plt.savefig('uncertainty_single.png', dpi=300, bbox_inches='tight')


# 示例数据
df1 = pd.DataFrame({
    'x': ["okvqa", "ocrvqa", "gqa", "vqav2", "textcaps", "a_okvqa", "refcoco", "llava_instruct", "sharegpt", "vg"],
    'y1': [56.50, 44.52, 34.74, 46.15, 27.85, 56.50, 41.81, 41.08, 39.18, 39.46],
    # 'y2': [57.26, 32.78, 18.21, 40.66, 15.70, 57.26, 34.40, 31.16, 31.56, 29.16],
    'y3': [27.48, 35.21, 36.43, 35.29, 7.46, 20.26, 20.08, 21.17, 20.65, 26.63]
})

df2 = pd.DataFrame({
    'x': ["okvqa", "ocrvqa", "gqa", "vqav2", "textcaps", "a_okvqa", "refcoco", "llava_instruct", "sharegpt", "vg"],
    'y3': [2.54, 3.45, 4.79, 2.92, 4.40, 2.54, 3.13, 3.34, 3.18, 3.43]
})

# 创建一个图形和第一个轴
fig, ax1 = plt.subplots()

# 在第一个轴上绘制前两个数据集
sns.lineplot(data=df1, x='x', y='y1', ax=ax1, label='Avg Acc', linewidth=2, color='green')
# sns.lineplot(data=df1, x='x', y='y2', ax=ax1, label='Avg UAcc', linewidth=2, color='orange')
sns.lineplot(data=df1, x='x', y='y3', ax=ax1, label='Test Acc', linewidth=2, color='grey')
ax1.set_ylabel('Accuracy')
ax1.tick_params(axis='y')
ax1.set_xlabel("Number of Models Merged")

# 创建第二个轴，共享x轴
ax2 = ax1.twinx()

# 在第二个轴上绘制第三个数据集
sns.lineplot(data=df2, x='x', y='y3', ax=ax2, label='Set Size', color='blue', linewidth=2)
ax2.set_ylabel('Set Size')
ax2.tick_params(axis='y')

# 调整图例位置
lines, labels = ax1.get_legend_handles_labels()
lines2, labels2 = ax2.get_legend_handles_labels()
ax2.legend(lines + lines2, labels + labels2)
ax1.legend_.remove()

# 旋转x轴标签
ax1.set_xticklabels(ax1.get_xticklabels(), rotation=45, horizontalalignment='right')

ax1.grid(True)
ax2.grid(False)  # 防止双轴情况下重复显示网格线

plt.savefig('modified_uncertainty_single.png', dpi=300, bbox_inches='tight')
plt.show()
