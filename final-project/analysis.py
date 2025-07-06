import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

df = pd.read_csv('build/benchmark_topkn_20250706_162403_thread12.csv')
print(df)
df = df[df['method'] != 'method']
df['data_size'] = df['data_size'].astype(int)
df['time(us)'] = df['time(us)'].astype(float)
methods = df['method'].unique()
sizes = df['data_size'].unique()
bar_width = 0.8 / len(methods)
x_base = np.arange(len(sizes))

colors = plt.cm.tab10.colors if len(methods) <= 10 else plt.cm.tab20.colors

plt.figure(figsize=(14, 7))

for idx, method in enumerate(methods):
    times = []
    for size in sizes:
        row = df[(df['data_size'] == size) & (df['method'] == method)]
        if not row.empty:
            times.append(row['time(us)'].values[0])
        else:
            times.append(np.nan)
    x = x_base + idx * bar_width
    plt.bar(
        x, times, width=bar_width, label=method,
        color=colors[idx % len(colors)],
        edgecolor='black', linewidth=1.2, alpha=0.95
    )

plt.xlabel('data_size', fontsize=18)
plt.ylabel('time (us)', fontsize=18)
plt.title('All Topkn Algorithms Performance Comparison', fontsize=28, fontweight='bold', pad=15)
plt.xticks(x_base + bar_width * (len(methods) - 1) / 2, [str(s) for s in sizes], fontsize=14)
plt.yticks(fontsize=16)
plt.legend(fontsize=16, loc='upper left', frameon=False)
plt.grid(axis='y', linestyle=':', linewidth=0.7, alpha=0.5)
plt.tight_layout(pad=2)
plt.yscale('log')
plt.savefig('all_algorithms_bar_simple.png', dpi=160)
plt.show()