import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

result_dir = '_fastflow_experiment_checkpoints/exp4'
accuracy_df = pd.read_csv(f'{result_dir}/results.csv')  # 假设文件名为accuracy_record.csv
loss_df = pd.read_csv(f'{result_dir}/losses.csv')

plt.figure(figsize=(10, 5))
plt.subplot(1, 2, 1)  # 创建一个1x2的子图布局
plt.plot(accuracy_df['Epoch'], accuracy_df['Pixel-AUROC'], label='Pixel-AUROC')
plt.plot(accuracy_df['Epoch'], accuracy_df['Image-AUROC'], label='Image-AUROC')
plt.title('AUROC per Epoch')
plt.xlabel('Epoch')
plt.ylabel('AUROC')
plt.grid(True)
plt.legend()

# 绘制损失值的折线图
plt.subplot(1, 2, 2)
plt.plot(loss_df['Epoch'], loss_df['Loss_Val'], label='Loss')
plt.title('Loss per Epoch')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.grid(True)
plt.legend()

# 调整子图布局
plt.tight_layout()

# 保存图表
plt.savefig(f'{result_dir}training_results.png')  # 保存为PNG格式的图片

# 如果需要，也可以显示图表
plt.show()