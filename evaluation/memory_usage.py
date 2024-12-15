import matplotlib.pyplot as plt
import os
import tracemalloc

def measure_memory_usage(func, *args, **kwargs):
    """
    Measure memory usage of a function using tracemalloc.

    This function wraps around any given function `func` and measures its 
    peak memory usage during execution. It utilizes Python's `tracemalloc` 
    library for tracking memory allocations.

    Params:
        - func: The function whose memory usage is to be measured.
        - *args: Positional arguments for the function `func`.
        - **kwargs: Keyword arguments for the function `func`.

    Returns:
        - Peak memory usage in megabytes (MB).
    """
    tracemalloc.start()

    #run the function with the given arguments
    func(*args, **kwargs)

    #get the current and peak memory usage
    current, peak = tracemalloc.get_traced_memory()

    #stop tracking memory allocations
    tracemalloc.stop()

    #convert peak memory usage from bytes to megabytes and return
    return peak / (1024 * 1024)  # Convert to MB


def plot_memory_usage(memory_usage_kmeans, memory_usage_srp):
    """
    Plot a horizontal bar chart comparing memory usage of K-means LSH and SRP-LSH.

    This function takes the measured memory usage of two methods and visualizes 
    the comparison as a horizontal bar chart using Matplotlib.

    Params:
        - memory_usage_kmeans: Memory usage of K-means LSH in MB.
        - memory_usage_srp: Memory usage of SRP-LSH in MB.
    """
    methods = ["K-means LSH", "SRP-LSH"]
    memory_usages = [memory_usage_kmeans, memory_usage_srp]

    plt.figure(figsize=(8, 5))

    #create the horizontal bar chart for analysis
    plt.barh(methods, memory_usages, color=["blue", "orange"], alpha=0.7)

    #add labels and a title to the plot
    plt.xlabel("Memory Usage (MB)", fontsize=12)
    plt.title("Memory Usage Comparison: K-means LSH vs. SRP-LSH", fontsize=14)

    #annotate the bars with the memory usage values
    for i, usage in enumerate(memory_usages):
        plt.text(usage + 2, i, f"{usage:.2f} MB", va="center", fontsize=10)

    plt.grid(axis="x", linestyle="--", alpha=0.6)
    plt.tight_layout()

    #Setting output directory to be comparison_plots
    comparison_plots_dir = os.path.join(os.getcwd(), "plots/comparison_plots") 
    plt.savefig(os.path.join(comparison_plots_dir, "plot_memory_usage.png"))
    plt.show()
