import time
import random
import math
import os
import csv
import argparse
from dataclasses import dataclass
from typing import List, Tuple, Optional

# Check if plotting libraries are available
try:
    import numpy as np
    import matplotlib.pyplot as plt

    _HAS_PLOTTING = True
except ImportError:
    _HAS_PLOTTING = False
    print(
        "Warning: numpy or matplotlib not found. Experiments will run but no plots will be generated."
    )
    print("Please install: conda install numpy matplotlib")


# Data structures
@dataclass
class Point:
    """
    Represents a restaurant or point of interest (POI).
    Using @dataclass (Python 3.7+ standard library) to auto-generate __init__.
    """

    x: float
    y: float
    name: str  # e.g., "Taco Bell"


class KDTreeNode:
    """Node in a k-d tree"""

    def __init__(self, point: Point, axis: int, left=None, right=None):
        self.point = point  # The point stored at this node
        self.axis = axis  # Splitting axis (0 for x, 1 for y)
        self.left = left
        self.right = right


# ----------------------- Range Reporting Algorithms -----------------------


# Algorithm A (Efficient): k-d Tree with Divide and Conquer
def build_kdtree(points: List[Point], depth: int = 0) -> Optional[KDTreeNode]:
    """
    Build a k-d tree recursively using divide-and-conquer.
    Time Complexity: T(n) = 2T(n/2) + O(n) -> O(n log n)

    Args:
        points: List of points to build the tree from
        depth: Current depth in the tree (determines splitting axis)

    Returns:
        Root node of the k-d tree
    """
    if not points:
        return None

    # 1. Divide: Choose splitting axis based on depth (0=x, 1=y)
    axis = depth % 2

    # 2. Conquer: Find the median point
    #    For O(n log n) construction, we could use O(n) quickselect to find median
    #    But for simplicity, O(n log n) sorting works fine, overall complexity stays same
    points.sort(key=lambda p: p.x if axis == 0 else p.y)
    median_idx = len(points) // 2
    median_point = points[median_idx]

    # 3. Combine: Create node and recursively build left and right subtrees
    return KDTreeNode(
        point=median_point,
        axis=axis,
        left=build_kdtree(points[:median_idx], depth + 1),
        right=build_kdtree(points[median_idx + 1 :], depth + 1),
    )


def query_kdtree(node: KDTreeNode, rect: Tuple, depth: int = 0) -> List[Point]:
    """
    Query a rectangular range in the k-d tree using divide-and-conquer.
    Time Complexity:
        Average: O(log n + k), where k is the number of points reported
        Worst: O(sqrt(n) + k)

    Args:
        node: Current node in the k-d tree
        rect: Query rectangle (x1, y1, x2, y2)
        depth: Current depth (for debugging purposes)

    Returns:
        List of points within the query rectangle
    """
    if node is None:
        return []

    (x1, y1, x2, y2) = rect
    axis = node.axis
    results = []

    # 1. Check if current node is within the rectangle
    if x1 <= node.point.x <= x2 and y1 <= node.point.y <= y2:
        results.append(node.point)

    # 2. Divide and Conquer: Decide whether to search subtrees (pruning)
    #    Get the splitting value at current node
    pivot_val = node.point.x if axis == 0 else node.point.y
    rect_min = x1 if axis == 0 else y1
    rect_max = x2 if axis == 0 else y2

    # Search left/bottom subtree
    if rect_min < pivot_val:
        results.extend(query_kdtree(node.left, rect, depth + 1))

    # Search right/top subtree
    if rect_max >= pivot_val:
        results.extend(query_kdtree(node.right, rect, depth + 1))

    return results


# Algorithm B (Baseline): Naive Scan
def report_naive(points: List[Point], rect: Tuple) -> List[Point]:
    """
    Naive O(n) linear scan algorithm for comparison.
    Simply checks every point against the query rectangle.

    Args:
        points: List of all points
        rect: Query rectangle (x1, y1, x2, y2)

    Returns:
        List of points within the query rectangle
    """
    (x1, y1, x2, y2) = rect
    result = []
    for p in points:
        if x1 <= p.x <= x2 and y1 <= p.y <= y2:
            result.append(p)
    return result


# ----------------------- Experimental Framework -----------------------


def generate_points(n: int, max_coord: int = 10000) -> List[Point]:
    """
    Generate random restaurant/POI data for testing.

    Args:
        n: Number of points to generate
        max_coord: Maximum coordinate value

    Returns:
        List of randomly generated points
    """
    return [
        Point(
            x=random.uniform(0, max_coord),
            y=random.uniform(0, max_coord),
            name=f"Restaurant_{i}",
        )
        for i in range(n)
    ]


def run_complexity_experiment(
    ns: List[int], trials: int = 3, out_dir: str = "experiments"
):
    """
    Run complexity experiments to measure build and query times.
    Saves results to CSV and generates plots.

    Args:
        ns: List of n values to test
        trials: Number of trials per n value
        out_dir: Output directory for results
    """
    os.makedirs(out_dir, exist_ok=True)

    # Store results (algorithm, n, mean_time, std_dev)
    results_build = []
    results_query_kdtree = []
    results_query_naive = []

    for n in ns:
        print(f"\n--- Testing N = {n} (trials={trials}) ---")

        build_times = []
        query_kdtree_times = []
        query_naive_times = []

        for t in range(trials):
            # 1. Generate test data
            points = generate_points(n)

            # 2. Measure Build time (k-d Tree)
            t0 = time.perf_counter()
            kdtree = build_kdtree(
                points.copy()
            )  # .copy() is important since build sorts
            t1 = time.perf_counter()
            build_times.append(t1 - t0)

            # 3. Generate query rectangles
            query_rects = []
            for _ in range(100):  # 100 query rectangles per N
                x_start = random.uniform(0, 9000)
                y_start = random.uniform(0, 9000)
                query_rects.append(
                    (x_start, y_start, x_start + 1000, y_start + 1000)
                )  # Query 10% of area

            # 4. Measure Query time (k-d Tree)
            t_kdtree_total = 0
            for rect in query_rects:
                tq0 = time.perf_counter()
                query_kdtree(kdtree, rect)
                tq1 = time.perf_counter()
                t_kdtree_total += tq1 - tq0
            query_kdtree_times.append(
                t_kdtree_total / len(query_rects)
            )  # Average query time

            # 5. Measure Query time (Naive)
            #    Only run on smaller N, as it becomes too slow otherwise
            if n <= 20000:
                t_naive_total = 0
                for rect in query_rects:
                    tq_n0 = time.perf_counter()
                    report_naive(points, rect)
                    tq_n1 = time.perf_counter()
                    t_naive_total += tq_n1 - tq_n0
                query_naive_times.append(t_naive_total / len(query_rects))

        # Calculate statistics
        mean_build = np.mean(build_times)
        std_build = np.std(build_times, ddof=0)
        results_build.append(("kdtree_build", n, mean_build, std_build))
        print(f"[k-d Tree Build] n={n} mean={mean_build:.6f}s std={std_build:.6f}s")

        mean_q_kdtree = np.mean(query_kdtree_times)
        std_q_kdtree = np.std(query_kdtree_times, ddof=0)
        results_query_kdtree.append(("kdtree_query", n, mean_q_kdtree, std_q_kdtree))
        print(
            f"[k-d Tree Query] n={n} mean={mean_q_kdtree:.8f}s std={std_q_kdtree:.8f}s"
        )

        if query_naive_times:
            mean_q_naive = np.mean(query_naive_times)
            std_q_naive = np.std(query_naive_times, ddof=0)
            results_query_naive.append(("naive_query", n, mean_q_naive, std_q_naive))
            print(
                f"[Naive Query]    n={n} mean={mean_q_naive:.6f}s std={std_q_naive:.6f}s"
            )

    # Write results to CSV
    csv_path = os.path.join(out_dir, "dc_timing.csv")
    with open(csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["variant", "n", "mean_s", "std_s"])
        for r in results_build + results_query_kdtree + results_query_naive:
            writer.writerow([r[0], r[1], f"{r[2]:.8f}", f"{r[3]:.8f}"])
    print(f"\nExperimental data saved to: {csv_path}")

    # Generate plots
    if not _HAS_PLOTTING:
        print("matplotlib/numpy not installed, skipping plots.")
        return

    try:
        # Plot 1: Build time
        plt.figure(figsize=(10, 6))
        ns_build = [r[1] for r in results_build]
        means_build = [r[2] for r in results_build]
        stds_build = [r[3] for r in results_build]

        plt.errorbar(
            ns_build,
            means_build,
            yerr=stds_build,
            fmt="o-",
            label="k-d Tree Build (Empirical)",
        )

        # Fit O(n log n) theoretical curve
        if ns_build:
            xs_log = np.array([n * math.log(max(n, 2)) for n in ns_build])
            ys = np.array(means_build)
            c_build = float(np.sum(xs_log * ys) / np.sum(xs_log * xs_log))
            model_vals = c_build * xs_log
            plt.plot(
                ns_build,
                model_vals,
                "k:",
                label=f"Theoretical $O(n \log n)$ (c={c_build:.2e})",
            )

        plt.xlabel("n (number of points)")
        plt.ylabel("Average build time (seconds)")
        plt.title("k-d Tree Build Time (Divide and Conquer)")
        plt.legend()
        plt.grid(True)
        build_plot_path = os.path.join(out_dir, "dc_build_time.png")
        plt.savefig(build_plot_path, dpi=150)
        plt.close()
        print(f"Build plot saved: {build_plot_path}")

        # Plot 2: Query time (k-d Tree vs Naive)
        plt.figure(figsize=(10, 6))

        # k-d Tree Query (blue line)
        ns_kdtree_q = [r[1] for r in results_query_kdtree]
        means_kdtree_q = [r[2] for r in results_query_kdtree]
        stds_kdtree_q = [r[3] for r in results_query_kdtree]
        plt.errorbar(
            ns_kdtree_q,
            means_kdtree_q,
            yerr=stds_kdtree_q,
            fmt="o-",
            label="k-d Tree Query (Empirical)",
        )

        # Naive Query (orange line)
        ns_naive_q = [r[1] for r in results_query_naive]
        means_naive_q = [r[2] for r in results_query_naive]
        stds_naive_q = [r[3] for r in results_query_naive]
        if ns_naive_q:
            plt.errorbar(
                ns_naive_q,
                means_naive_q,
                yerr=stds_naive_q,
                fmt="s--",
                label="Naive Scan (Empirical O(n))",
            )

        plt.xlabel("n (number of points)")
        plt.ylabel("Average query time (seconds)")
        plt.title("k-d Tree vs Naive Query Time")
        plt.legend()
        plt.grid(True)
        # Use log scale on Y-axis to better show the difference between O(log n) and O(n)
        plt.yscale("log")

        query_plot_path = os.path.join(out_dir, "dc_query_vs_naive.png")
        plt.savefig(query_plot_path, dpi=150)
        plt.close()
        print(f"Query plot saved: {query_plot_path}")

    except Exception as e:
        print(f"Error during plotting: {e}")


def main():
    parser = argparse.ArgumentParser(description="k-d Tree range reporting experiments")
    parser.add_argument(
        "--run-exp",
        action="store_true",
        default=True,
        help="Run complexity experiment and generate plots",
    )
    parser.add_argument(
        "--example", action="store_true", help="Run a small example and print results"
    )
    args = parser.parse_args()

    if args.example:
        print("Running small example...")
        points = [
            Point(2, 3, "A"),
            Point(5, 4, "B"),
            Point(9, 6, "C"),
            Point(4, 7, "D"),
            Point(8, 1, "E"),
            Point(7, 2, "F"),
        ]
        kdtree = build_kdtree(points.copy())
        rect = (4, 1, 8, 5)  # Query range [4,8] x [1,5]

        print(f"Query rectangle: {rect}")
        results_kdtree = query_kdtree(kdtree, rect)
        print("[k-d Tree Results]:")
        for p in results_kdtree:
            print(f"  {p.name} @ ({p.x}, {p.y})")

        results_naive = report_naive(points, rect)
        print("\n[Naive Results]:")
        for p in results_naive:
            print(f"  {p.name} @ ({p.x}, {p.y})")
        return

    if args.run_exp:
        # Test on larger scales to demonstrate scalability
        # Note: Naive algorithm is only tested up to n=20K as it becomes too slow
        ns = [1000, 5000, 10000, 20000, 50000, 100000, 200000, 500000]
        run_complexity_experiment(ns, trials=3)


if __name__ == "__main__":
    random.seed(42)  # Ensure reproducible experiments
    main()
