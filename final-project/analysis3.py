import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

df = pd.read_csv('build/benchmark_topkn_20250706_162403_thread12.csv')
df = df[df['method'] != 'method']
df['data_size'] = df['data_size'].astype(int)
df['time(us)'] = df['time(us)'].astype(float)
methods = df['method'].unique()
sizes = np.sort(df['data_size'].unique())

plt.figure(figsize=(12, 7))
colors = plt.cm.tab10.colors if len(methods) <= 10 else plt.cm.tab20.colors

for idx, method in enumerate(methods):
    times = []
    for size in sizes:
        row = df[(df['data_size'] == size) & (df['method'] == method)]
        times.append(row['time(us)'].values[0] if not row.empty else np.nan)
    plt.plot(sizes, times, marker='o', label=method, color=colors[idx % len(colors)], linewidth=2)
    # 数据点标记
    for x, y in zip(sizes, times):
        plt.text(x, y, f"{int(y)}", fontsize=9, ha='center', va='bottom')

plt.xscale('log')
plt.yscale('log')
plt.xlabel('data_size', fontsize=16)
plt.ylabel('time (us)', fontsize=16)
plt.title('All Topkn Algorithms Performance Comparison', fontsize=22, fontweight='bold', pad=15)
plt.xticks(sizes, [str(s) for s in sizes], fontsize=12)
plt.yticks(fontsize=12)
plt.legend(fontsize=13, loc='upper left', frameon=False)
plt.grid(True, which="both", ls=":", linewidth=0.7, alpha=0.5)
plt.tight_layout(pad=2)
plt.savefig('all_algorithms_line.png', dpi=160)
plt.show()