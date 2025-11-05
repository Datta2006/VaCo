# MBAT Optimizer for Vehicle-Assisted Service Caching (VaCo)

This project implements the **Multi-Swarm Collaborative Bat Algorithm (MBAT)** to solve the joint optimization problem of **task offloading** and **service caching** in a **Vehicle-Assisted Vehicular Edge Computing (VEC)** network.

The core objective (inspired by *“Vehicle-Assisted Service Caching for Task Offloading in Vehicular Edge Computing”*) is to minimize a **System Efficiency** function — a weighted sum of the **Service Failure Rate** and the **System Cost**.

---

## 🚗 VaCo System Model

### Components

| Component             | Description                                             |
| --------------------- | ------------------------------------------------------- |
| **VEC Servers (J)**   | Roadside units with limited storage & compute resources |
| **Vehicles (M)**      | Mobile users generating tasks & caching content         |
| **Tasks (I)**         | Computation-intensive workloads                         |
| **Content Types (K)** | Different service content types                         |
| **MBAT Optimizer**    | Optimizes offloading & caching decisions                |

### Decision Variables

| Variable | Shape      | Meaning                                         |
| -------- | ---------- | ----------------------------------------------- |
| **P1**   | (Np, I)    | Task offloading: `1 = Local`, `2..J+1 = Server` |
| **P2**   | (Np, J, K) | Server caching matrix (binary)                  |
| **P3**   | (Np, I)    | Content retrieval: `1 = Vehicle`, `0 = Server`  |

---

## 🎯 Optimization Objective

Minimize:

System Efficiency = ω × Cost  +  (1 − ω) × Failure_Rate

* **Cost** — Usage of VEC + vehicle resources
* **Failure Rate** — Tasks violating latency constraints
* **ω** dynamically changes for balanced exploration (Pareto-based)

---

## 🦇 MBAT Algorithm Overview

| Feature                        | Description                                  |
| ------------------------------ | -------------------------------------------- |
| **Multi-Swarm**                | P1, P2, P3 optimized as collaborative swarms |
| **Global Collaboration**       | All swarms move toward the global-best       |
| **Exploration → Exploitation** | Loudness & Pulse Rate adaptation             |
| **Guided Mutation**            | Moves solutions toward global best           |
| **Local Search**               | Maintains diversity (binary flips, shifts)   |
| **Implicit Pareto Awareness**  | Multi-objective balance during selection     |

---

## ▶️ How to Run

### Requirements

```
pip install numpy matplotlib
```

### Execute

Save code as `mbat_optimizer.py`
Then run:

```
python mbat_optimizer.py
```

---

## 📈 Expected Output

### Terminal Logs

* Iteration count
* Best Efficiency, Cost, Failure Rate

### Plots Produced

* Best & average system efficiency convergence
* Cost convergence
* Failure-rate convergence
* Pareto front (Cost vs Failure Rate)
* Task allocation & content retrieval bar graph

### Printed Solution

* Best System Efficiency
* Final `P1`, `P2`, `P3` matrices

---

## 🧠 Result Interpretation

| Outcome                    | Meaning                                      |
| -------------------------- | -------------------------------------------- |
| **Low Cost, High Failure** | Mostly local execution, minimal caching      |
| **High Cost, Low Failure** | Heavy server offloading + aggressive caching |
| **Balanced Optimal**       | Pareto-optimal mix found by MBAT             |

Goal: Achieve lowest **System Efficiency** score by balancing cost & failure risk.

---

## 📚 Citation

If using this for research:

*Vehicle-Assisted Service Caching for Task Offloading in Vehicular Edge Computing*

---

## ⭐ Contribution

Pull requests are welcome — feel free to improve the MBAT model or VaCo simulation!

---

Let me know if you want a **shorter version**, **IEEE-style abstract**, or **LaTeX README** for GitHub/Thesis!
