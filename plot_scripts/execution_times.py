import matplotlib.pyplot as plt

# Data
implementations = ['Sequential', 'Parallel 1T', 'Parallel 2T', 'Parallel 4T', 'Parallel 8T', 'Parallel 16T', 'Parallel 20T', 'Parallel 40T']
operations = ['Gaussian Blur', 'Edge Detection', 'Grayscale Conversion']
execution_times = [
    # [0.028463, 0.698067, 0.0328859],
    [58.096, 9.20468, 4.17621],
    [58.0018, 11.0401, 3.57148],
    [29.4638, 5.70941, 1.86815],
    [15.7054, 3.18815, 0.986805],
    [8.13709, 1.83999, 0.643133],
    [4.53652, 0.990168, 0.264056],
    [3.97512, 0.869599, 0.234472],
    [3.39541, 0.895125, 0.186632]
]

# Plot
bar_width = 0.2
index = range(len(implementations))

for i, operation in enumerate(operations):
    plt.bar([x + i * bar_width for x in index], [row[i] for row in execution_times], bar_width, label=operation)
    for j, val in enumerate(execution_times):
        plt.text(j + i * bar_width - 0.05, val[i] + 0.5, f'{val[i]:.2f}s', color='black')

plt.xlabel('Implementations')
plt.ylabel('Execution Time (seconds)')
plt.title('Execution Time for Different Implementations and Operations')
plt.xticks([r + bar_width for r in range(len(implementations))], implementations)
plt.legend()

plt.tight_layout()
plt.show()
