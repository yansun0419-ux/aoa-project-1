
from dataclasses import dataclass
from typing import List, Tuple, Optional
import random
import time
import math
import csv
import os
import argparse

try:
    import numpy as np
    import matplotlib.pyplot as plt
    _HAS_PLOTTING = True
except Exception:
    _HAS_PLOTTING = False

@dataclass
class NewsItem:# Data model
    id: int
    deadline: int
    value: float
    decay: float = 0.0
    duration: int = 1

def make_parent(m: int) -> List[int]: #DSU Implement
    return list(range(m + 1))

def find_parent(parent: List[int], x: int) -> int:
    if x < 0:
        return 0
    while parent[x] != x:
        parent[x] = parent[parent[x]]
        x = parent[x]
    return x

def union_parent(parent: List[int], u: int, v: int) -> None:
    ru = find_parent(parent, u)
    rv = find_parent(parent, v)
    parent[ru] = rv


#----------------------- NewsItem Scheduling two algorithms(native and greedy) -----------------------
def schedule_news(items: List[NewsItem]) -> Tuple[List[Tuple[int, NewsItem]], float]: #greedy algrithm
    if not items:
        return [], 0.0
    # clamp deadlines to n to bound M
    n = len(items)
    for it in items:
        if it.deadline > n:
            it.deadline = n
    sorted_items = sorted(items, key=lambda x: x.value, reverse=True)
    M = max(it.deadline for it in items)
    parent = make_parent(M)
    schedule: List[Tuple[int, NewsItem]] = []
    total = 0.0
    for it in sorted_items:
        d = min(it.deadline, M)
        s = find_parent(parent, d)
        if s > 0:
            schedule.append((s, it))
            total += it.value
            union_parent(parent, s, s - 1)
    schedule.sort(key=lambda x: x[0])
    return schedule, total

def schedule_news_naive(items: List[NewsItem]) -> Tuple[List[Tuple[int, NewsItem]], float]: #brute force algorithm
    if not items:
        return [], 0.0
    n = len(items)
    for it in items:
        if it.deadline > n:
            it.deadline = n
    sorted_items = sorted(items, key=lambda x: x.value, reverse=True)
    M = max(it.deadline for it in items)
    capacity = [1] * (M + 1)  # 1 per slot
    schedule: List[Tuple[int, NewsItem]] = []
    total = 0.0
    for it in sorted_items:
        for s in range(min(it.deadline, M), 0, -1):
            if capacity[s] > 0:
                capacity[s] -= 1
                schedule.append((s, it))
                total += it.value
                break
    schedule.sort(key=lambda x: x[0])
    return schedule, total


def generate_realistic_campus_events(n: int, max_slots: int = 1000) -> List[NewsItem]:# Random Data generation
    items: List[NewsItem] = []
    for i in range(1, n + 1):
        value = random.paretovariate(1.5)
        if random.random() < 0.7:
            deadline = random.randint(1, min(10, max_slots))
        else:
            deadline = random.randint(1, max_slots)
        decay = random.uniform(0.0, 1.0)
        items.append(NewsItem(id=i, deadline=deadline, value=value, decay=decay))
    return items



# Experimental test
def run_complexity_experiment(schedule_fn_dsu, schedule_fn_naive,
                                ns: List[int], trials: int = 3, out_dir: str = "experiments"):
    os.makedirs(out_dir, exist_ok=True)
    measured_dsu = []
    measured_naive = []

    for n in ns:
        # DSU timing
        times = []
        for _ in range(trials):
            items = generate_realistic_campus_events(n, max_slots=max(1000, int(n//10)))
            t0 = time.perf_counter()
            schedule_fn_dsu([NewsItem(i+1, it.deadline, it.value, it.decay) for i, it in enumerate(items)])
            t1 = time.perf_counter()
            times.append(t1 - t0)
        mean = float(np.mean(times)) if _HAS_PLOTTING else (sum(times) / len(times))
        std = float(np.std(times, ddof=0)) if _HAS_PLOTTING else ((max(times) - min(times)) / 2.0)
        measured_dsu.append((n, mean, std))
        print(f"[DSU] n={n} mean={mean:.6f}s std={std:.6f}s")

        # Naive timing /Brute-force algorithm
        if n <= 50000:
            times_n = []
            for _ in range(trials):
                items = generate_realistic_campus_events(n, max_slots=max(1000, int(n//10)))
                t0 = time.perf_counter()
                schedule_fn_naive([NewsItem(i+1, it.deadline, it.value, it.decay) for i, it in enumerate(items)])
                t1 = time.perf_counter()
                times_n.append(t1 - t0)
            mean_n = float(np.mean(times_n)) if _HAS_PLOTTING else (sum(times_n) / len(times_n))
            std_n = float(np.std(times_n, ddof=0)) if _HAS_PLOTTING else ((max(times_n) - min(times_n)) / 2.0)
            measured_naive.append((n, mean_n, std_n))
            print(f"[NAIVE] n={n} mean={mean_n:.6f}s std={std_n:.6f}s")
        else:
            sample_n = 50000
            times_n = []
            for _ in range(trials):
                items = generate_realistic_campus_events(sample_n, max_slots=max(1000, int(sample_n//10)))
                t0 = time.perf_counter()
                schedule_fn_naive([NewsItem(i+1, it.deadline, it.value, it.decay) for i, it in enumerate(items)])
                t1 = time.perf_counter()
                times_n.append(t1 - t0)
            mean_n = float(np.mean(times_n)) if _HAS_PLOTTING else (sum(times_n) / len(times_n))
            std_n = float(np.std(times_n, ddof=0)) if _HAS_PLOTTING else ((max(times_n) - min(times_n)) / 2.0)
            measured_naive.append((n, mean_n, std_n))
            print(f"[NAIVE-sampled] n={n} (sample {sample_n}) mean~{mean_n:.6f}s std~{std_n:.6f}s")

    # Fit DSU model mean ~ c * n log n
    xs = np.array([n * math.log(max(n, 2)) for n, _, _ in measured_dsu]) if _HAS_PLOTTING else []
    ys = np.array([mean for _, mean, _ in measured_dsu]) if _HAS_PLOTTING else []
    c = float(np.sum(xs * ys) / np.sum(xs * xs)) if (_HAS_PLOTTING and len(xs) > 0) else 0.0
    print("Fitted factor c:", c)

    csv_path = os.path.join(out_dir, "greedy_timing.csv")
    with open(csv_path, "w", newline='') as f:
        writer = csv.writer(f)
        writer.writerow(["variant", "n", "mean_s", "std_s", "model_c"])
        for (n, mean, std) in measured_dsu:
            writer.writerow(["dsu", n, f"{mean:.6f}", f"{std:.6f}", f"{c:.6e}"])
        for (n, mean, std) in measured_naive:
            writer.writerow(["naive", n, f"{mean:.6f}", f"{std:.6f}", ""])
    print("Timing CSV written to:", csv_path)

    # Make Picture
    if not _HAS_PLOTTING:
        print("numpy/matplotlib not available; skipping plots.")
        return csv_path

    try:
        ns_dsu = [n for n, _, _ in measured_dsu]
        means_dsu = [m for _, m, _ in measured_dsu]
        stds_dsu = [s for _, _, s in measured_dsu]
        ns_naive = [n for n, _, _ in measured_naive]
        means_naive = [m for _, m, _ in measured_naive]
        stds_naive = [s for _, _, s in measured_naive]

        plt.figure(figsize=(8, 5))
        if ns_dsu:
            plt.errorbar(ns_dsu, means_dsu, yerr=stds_dsu, fmt='o-', label='DSU empirical')
        if ns_naive:
            plt.errorbar(ns_naive, means_naive, yerr=stds_naive, fmt='s--', label='Naive empirical')
        if ns_dsu and c > 0:
            xs_model = np.array([n * math.log(max(n, 2)) for n in ns_dsu])
            model_vals = c * xs_model
            plt.plot(ns_dsu, model_vals, 'k:', label=f'c * n log n (c={c:.2e})')
        plt.xlabel('n (number of NewsItems)')
        plt.ylabel('avg runtime (s)')
        plt.title('Empirical runtimes and fitted theoretical curve')
        plt.legend()
        plt.grid(True)
        p_main = os.path.join(out_dir, "greedy_vs_naive_and_theory.png")
        plt.savefig(p_main, dpi=150)
        plt.close()

        if ns_dsu:
            norm_vals = [mean / (n * math.log(max(n, 2))) for n, mean, _ in measured_dsu]
            plt.figure(figsize=(8, 4))
            plt.plot(ns_dsu, norm_vals, 'o-')
            plt.xlabel('n')
            plt.ylabel('time / (n log n)')
            plt.title('Normalized time: empirical time divided by n log n')
            plt.grid(True)
            p_norm = os.path.join(out_dir, "greedy_normalized.png")
            plt.savefig(p_norm, dpi=150)
            plt.close()

        if ns_dsu:
            plt.figure(figsize=(8, 5))
            plt.loglog(ns_dsu, means_dsu, 'o-', basex=10)
            if ns_naive:
                plt.loglog(ns_naive, means_naive, 's--')
            plt.xlabel('n (log scale)')
            plt.ylabel('avg runtime (log scale)')
            plt.title('Log-Log empirical runtime')
            plt.grid(True, which="both", ls="--")
            p_loglog = os.path.join(out_dir, "greedy_loglog.png")
            plt.savefig(p_loglog, dpi=150)
            plt.close()
        print("Plots saved:", p_main, os.path.join(out_dir, "greedy_normalized.png"), os.path.join(out_dir, "greedy_loglog.png"))
    except Exception as e:
        print("Error during plotting:", e)
        print("CSV is available at:", csv_path)

    return csv_path


def plot_from_csv(csv_path: str, out_dir: str = "experiments") -> None: #Read CSV and plot
    os.makedirs(out_dir, exist_ok=True)
    rows = []
    with open(csv_path, newline='') as f:
        reader = csv.DictReader(f)
        for r in reader:
            variant = r['variant']
            n = int(r['n'])
            mean = float(r['mean_s'])
            std = float(r['std_s'])
            model_c = r.get('model_c', '')
            rows.append((variant, n, mean, std, model_c))

    ds = [r for r in rows if r[0] == 'dsu']
    nv = [r for r in rows if r[0] == 'naive']

    if _HAS_PLOTTING:
        ns_dsu = [r[1] for r in ds]
        means_dsu = [r[2] for r in ds]
        stds_dsu = [r[3] for r in ds]
        plt.figure(figsize=(8, 5))
        plt.errorbar(ns_dsu, means_dsu, yerr=stds_dsu, fmt='o-', label='DSU empirical (CSV)')
        plt.xlabel('n'); plt.ylabel('avg runtime (s)')
        plt.grid(True)
        plt.savefig(os.path.join(out_dir, "greedy_empirical_runtime_from_csv.png"), dpi=150)
        plt.close()

        ns_naive = [r[1] for r in nv]
        means_naive = [r[2] for r in nv]
        stds_naive = [r[3] for r in nv]
        plt.figure(figsize=(8, 5))
        plt.errorbar(ns_naive, means_naive, yerr=stds_naive, fmt='s-', color='orange', label='Naive empirical (CSV)')
        plt.xlabel('n'); plt.ylabel('avg runtime (s)')
        plt.grid(True)
        plt.savefig(os.path.join(out_dir, "naive_empirical_runtime_from_csv.png"), dpi=150)
        plt.close()

        plt.figure(figsize=(8, 5))
        plt.plot(ns_dsu, means_dsu, 'o-', label='DSU empirical (CSV)')
        plt.plot(ns_naive, means_naive, 's--', label='Naive empirical (CSV)')
        plt.xlabel('n'); plt.ylabel('avg runtime (s)')
        plt.legend(); plt.grid(True)
        plt.savefig(os.path.join(out_dir, "greedy_vs_naive_from_csv.png"), dpi=150)
        plt.close()
        print("Plots generated from CSV into", out_dir)
    else:
        print("matplotlib/numpy not available; cannot plot from CSV.")


def main():
    parser = argparse.ArgumentParser(description="NewsItem scheduler experiments")
    parser.add_argument("--plot-complexity", action="store_true", help="Run complexity experiment and produce plots")
    parser.add_argument("--from-csv", type=str, help="Read timing CSV and generate plots from it")
    parser.add_argument("--example", action="store_true", help="Run example schedule and print")
    args = parser.parse_args()

    if args.example:
        items = [
            NewsItem(1, deadline=2, value=100.0),
            NewsItem(2, deadline=1, value=19.0),
            NewsItem(3, deadline=2, value=27.0),
            NewsItem(4, deadline=1, value=25.0),
            NewsItem(5, deadline=3, value=15.0),
        ]
        sch, tot = schedule_news(items)
        print("Scheduled (slot: id, deadline, value):")
        for s, it in sch:
            print(f"  slot {s}: id={it.id}, d={it.deadline}, v={it.value:.2f}")
        print("Total value:", tot)
        return

    if args.from_csv:
        plot_from_csv(args.from_csv)
        return

    # Default behavior: run complexity experiment (DSU + naive) and plot
    ns = [1000, 5000, 10000, 20000, 50000, 100000]
    run_complexity_experiment(schedule_news, schedule_news_naive, ns, trials=3)

if __name__ == "__main__":
    random.seed(42)
    main()