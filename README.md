# VaCo: Vehicle-assisted Collaborative Caching System Simulation

## 🌟 Project x

[cite\_start]This project implements and simulates **VaCo (Vehicle-assisted Collaborative caching system)**, a novel architecture for **Vehicular Edge Computing (VEC)** networks[cite: 7]. [cite\_start]The core goal of VaCo is to address the limited storage capacity of VEC servers by dynamically utilizing the caching resources of surrounding vehicles to support intelligent service content for task offloading[cite: 7].

[cite\_start]The simulation focuses on jointly optimizing the **Service Failure Rate** and the **Total Cost** in a highly dynamic VEC environment using a multi-objective optimization approach[cite: 11].

## 🚀 Key Features

  * [cite\_start]**Vehicle-Assisted Caching:** Allows VEC servers to download cached service content from any vehicle in the VEC network to support task offloading[cite: 8].
  * [cite\_start]**Optimization Algorithm:** Utilizes a custom **Multi-Swarm Collaborative Bat Algorithm (MBAT)** to solve the joint optimization problem of service caching and task offloading[cite: 86].
  * **Multi-Objective Optimization:** Jointly minimizes two conflicting objectives:
    1.  [cite\_start]**Service Failure Rate (F):** The proportion of tasks that exceed the maximum tolerable latency ($t_{max}$)[cite: 225, 227].
    2.  [cite\_start]**Total Cost (C):** Combines penalty cost for task failure and content access cost when invoking vehicle storage resources[cite: 233].
  * [cite\_start]**Dynamic Weight Index:** Uses a **Pareto-based optimization scheme** to design a dynamic weight index, evaluating the benefits of invoking vehicle resources and safeguarding the benefits of both vehicles and VEC servers simultaneously[cite: 11, 86].
  * [cite\_start]**Clustered Content Retrieval Mechanism:** Formulates the vehicle resource scheduling between VEC server clusters and vehicle clusters as a binary optimization problem[cite: 639].

## 🛠️ Technology Stack & Setup

This project primarily uses **Python** for implementing the core optimization algorithm, numerical operations, and generating initial performance plots.

### Prerequisites

  * **Python (3.8+):** Essential for running the MBAT algorithm (`MBATOptimizer` class), numerical computation (`numpy`), and plotting (`matplotlib`).
  * [cite\_start]**MATLAB (Optional/Planned):** May be required later for implementing and comparing advanced schemes like **NSGA-II** and **EHGSA**[cite: 449, 451].
  * [cite\_start]**Real-world Dataset:** The final version uses the **Roman taxis** mobility trace dataset for realistic scenario generation[cite: 433, 559].

### Current Simulation Parameters (MBATOptimizer Default)

> 💡 **Note:** These parameters are set within the Python code for initial convergence testing and may be updated to reflect real-world values later.

| **Parameter** | **Symbol** | **Value** | **Description** |
| :---: | :---: | :---: | :--- |
| VEC Servers | $J$ | `3` | [cite\_start]Number of roadside edge servers[cite: 427]. |
| Total Tasks | $I$ | `50` | [cite\_start]Number of computation tasks generated per time slot (initial setting)[cite: 431]. |
| Vehicles | $M$ | `100` | [cite\_start]Number of vehicles in the system (resource providers)[cite: 427]. |
| Content Types | $K$ | `10` | [cite\_start]Number of unique service contents available (initial setting)[cite: 427]. |
| Server Storage | $S_j$ | `1000 MB` | [cite\_start]Storage capacity per VEC server[cite: 428]. |
| Content Size | $s_k$ | `200 MB` | [cite\_start]Size of each content type (fixed)[cite: 430]. |

### Project Structure

```bash
VaCo-VEC-Simulation/
├── README.md
├── src/
│   ├── python/
│   │   ├── mbatt_optimizer.py  # ⚙️ MBATOptimizer class (Core implementation)
│   │   └── data_processor.py   # 🚗 Script to clean/format raw mobility data
│   ├── matlab/
│   │   ├── Main_Simulation.m   # 📊 Main script for running comparisons (Planned)
│   │   └── VaCo_Model.m        # 💻 Functions defining the VEC system model (Planned)
├── data/
│   └── raw/
│       └── roma_taxi_traces.csv # 🗺️ Original dataset files
└── results/
    ├── convergence_plots/
    └── performance_metrics.csv
```

## 📚 Core Methodology Summary

### 1\. **Decision Variables**

[cite\_start]The MBAT algorithm optimizes three coupled sets of decision variables[cite: 313]:

  * [cite\_start]**Task Offloading Decisions ($P_1$ / $x_{i}$):** Decides where task $i$ is executed[cite: 200].
  * [cite\_start]**Content Caching Decisions ($P_2$ / $z_{jk}$):** Binary matrix indicating if VEC server $j$ caches service content $k$[cite: 167].
  * [cite\_start]**Content Retrieval Decisions ($P_3$ / $y_{i}$):** Binary vector indicating if task $i$ attempts to retrieve required content from the vehicle cluster (1) or the server cluster (0)[cite: 177].

### 2\. **Objective Function**

[cite\_start]The goal is to minimize the optimization model, $\text{PO}$[cite: 251], by jointly optimizing the system efficiency:

$$\min_{x_{i}, z_{jk}, y_{i}} \omega C + (1 - \omega) F$$

Where:

  * [cite\_start]$F$ is the **Service Failure Rate**[cite: 231].
  * [cite\_start]$C$ is the **Total Cost**[cite: 244].
  * [cite\_start]$\omega$ is the dynamic weight factor optimized by the Pareto-based scheme to achieve an effective trade-off[cite: 266, 379].

### 3\. **Constraints**

> [cite\_start]🔒 **Storage Constraint:** The algorithm enforces the storage resource constraint of the VEC server (Eq. 19 from the paper)[cite: 267]:
> $$\sum_{k=1:K} z_{jk} s_{k}<S_{j}$$

## 📈 Analysis and Visualization

The Python environment is configured to generate real-time plots of the optimization process using `matplotlib`:

  * [cite\_start]**SMOOTH Convergence Progress:** Tracks the **Best** and **Average System Efficiency** over iterations[cite: 461].
  * [cite\_start]**Clear Pareto Front (Inverse Relationship):** Plots **Cost** vs. **Service Failure Rate** to visualize the fundamental trade-off of the multi-objective problem[cite: 593].
  * **Combined Objectives Convergence:** Shows the normalized convergence of both Cost and Failure Rate.
  * [cite\_start]**Final Solution Strategy:** Bar chart quantifying the distribution of task strategies (Offloading, Retrieval sources) in the final optimal solution[cite: 635].
