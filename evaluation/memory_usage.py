import matplotlib.pyplot as plt
import os
import tracemalloc

def measure_memory_usage(func, *args, **kwargs):
    """
    Measure memory usage of a function using tracemalloc.
    
    Params:
        - func: The function to measure.
        - *args, **kwargs: Arguments for the function.

    Returns:
        - Peak memory usage (in MB).
    """
    tracemalloc.start()
    func(*args, **kwargs)
    current, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    return peak / (1024 * 1024)  # Convert to MB


def plot_memory_usage(memory_usage_kmeans, memory_usage_srp):
    methods = ["K-means LSH", "SRP-LSH"]
    memory_usages = [memory_usage_kmeans, memory_usage_srp]

    plt.figure(figsize=(8, 5))
    plt.barh(methods, memory_usages, color=["blue", "orange"], alpha=0.7)
    plt.xlabel("Memory Usage (MB)", fontsize=12)
    plt.title("Memory Usage Comparison: K-means LSH vs. SRP-LSH", fontsize=14)
    for i, usage in enumerate(memory_usages):
        plt.text(usage + 2, i, f"{usage:.2f} MB", va="center", fontsize=10)
    plt.grid(axis="x", linestyle="--", alpha=0.6)
    plt.tight_layout()
    plt.show()
