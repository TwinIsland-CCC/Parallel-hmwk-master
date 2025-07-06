import pandas as pd
import matplotlib.pyplot as plt

# 读取数据
df = pd.read_csv('./build/benchmark_ccs_topk1_multicore.csv')

# 去除可能的多余逗号
df['data_size'] = df['data_size'].astype(int)
df['time(cycles)'] = df['time(cycles)'].astype(int)

# 按 method 分组画图
plt.figure(figsize=(8, 5))
for method, group in df.groupby('method'):
    plt.plot(group['data_size'], group['time(cycles)'], marker='o', label=method)
    # 添加数据标记
    for x, y in zip(group['data_size'], group['time(cycles)']):
        plt.text(x, y, str(y), fontsize=8, ha='right', va='bottom')

plt.xscale('log')
plt.yscale('log')
plt.xlabel('data_size')
plt.ylabel('time (cycles)')
plt.title('Top1 Task-level Multi-core Benchmark')
plt.legend()
plt.grid(True, which="both", ls="--", linewidth=0.5)
plt.tight_layout()
plt.show()