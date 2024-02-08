import matplotlib.pyplot as plt

# Data
operations = ['Gaussian Blur', 'Edge Detection', 'Grayscale Conversion']
sequential_times = [58.096, 9.20468, 4.17621]
parallel_times = [
    [58.0018, 29.4638, 15.7054, 8.13709, 4.53652, 3.97512, 3.39541],  # Gaussian Blur
    [11.0401, 5.70941, 3.18815, 1.83999, 0.990168, 0.869599, 0.895125],  # Edge Detection
    [3.57148, 1.86815, 0.986805, 0.643133, 0.264056, 0.234472, 0.186632],  # Grayscale Conversion
]

# Calculate speedup
speedup = [[seq_time / par_time for par_time in par_times] for seq_time, par_times in zip(sequential_times, parallel_times)]

# Plot
num_threads = [1, 2, 4, 8, 16, 20, 40]
plt.figure(figsize=(10, 6))

for i, operation in enumerate(operations):
    plt.plot(num_threads, speedup[i], marker='o', label=operation)

plt.xlabel('Number of Threads')
plt.ylabel('Speedup')
plt.title('Speedup vs. Number of Threads for Different Operations')
plt.xticks(num_threads)
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.show()
