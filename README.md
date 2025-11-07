# Analysis of Algorithms - Project 1
## Greedy and Divide-and-Conquer: Case Studies

This repository contains the full implementation, experimental data, and final report for the AOA Project 1 at the University of Florida.

**Group Members:**
* **Yingzhu Chen** (UFID: 83945095)
* **Forrest Yan Sun** (UFID: 47948015)

---

### Project Overview

This project implements and analyzes solutions for two problems, each corresponding to a major algorithmic paradigm as required by the assignment.

#### Part I: Greedy Algorithm (NewsItem Scheduling)
* **Problem:** Given a set of "NewsItems" (e.g., UF campus events), each with a value and a deadline, schedule them into limited time slots to maximize the total value.
* **Algorithm:** An optimal O(n log n) greedy algorithm using a Disjoint Set Union (DSU) data structure for efficient slot-finding.
* **Implementation:** `code/greedy-algorithm/news_scheduler.py`
* **Analysis:** Compared against a naive baseline, demonstrating the DSU's significant practical advantage.

#### Part II: Divide and Conquer (2D Range Reporting)
* **Problem:** Given a set of 2D points (e.g., restaurants in Gainesville), efficiently report all points that fall within a user-defined query rectangle.
* **Algorithm:** A **k-d Tree**, a classic divide-and-conquer spatial data structure.
* **Implementation:** `code/divide-and-conquer/range_reporting_kdtree.py`
* **Analysis:**
    * **Build:** O(n log² n) theoretical analysis (due to using `sort()` in each step) vs. a near-perfect O(n log n) empirical fit, demonstrating the small constant factor of Python's Timsort.
    * **Query:** O(log n + k) average-case query time is verified, showing a massive speedup over the O(n) naive scan.

---

### Repository Structure

```
/
├── code/
│   ├── greedy-algorithm/
│   │   └── news_scheduler.py       # Part I: Greedy DSU implementation
│   └── divide-and-conquer/
│       └── range_reporting_kdtree.py # Part II: D&C k-d Tree implementation
├── experiments/                    # Generated plots and CSV data
├── .gitignore
└── README.md                       # You are here
```

---

### How to Run Experiments

#### 1. Clone the repository:
```bash
git clone https://github.com/yansun0419-ux/aoa-project-1.git
cd aoa-project-1
```

#### 2. Install dependencies (preferably in a virtual environment):
```bash
pip install numpy matplotlib
```
   
#### 3. Run the Greedy Algorithm (Part I) experiments:

Run a small example:
```bash
python code/greedy-algorithm/news_scheduler.py --example
```

Run full complexity experiment (generates plots and CSV):
```bash
python code/greedy-algorithm/news_scheduler.py
```

#### 4. Run the Divide & Conquer (Part II) experiments:

Run a small example:
```bash
python code/divide-and-conquer/range_reporting_kdtree.py --example
```

Run full complexity experiment (generates plots and CSV):
```bash
python code/divide-and-conquer/range_reporting_kdtree.py
```

#### 5. View the results:
All generated plots and CSV data will be saved in the `experiments/` folder.
