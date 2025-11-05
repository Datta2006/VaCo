# VaCo – Vehicle-Assisted Service Caching Optimization

### Using Multi-Swarm Bat Algorithm (MBAT)

This project implements a **Multi-Swarm Bat Algorithm (MBAT)** to optimize **task offloading and service caching** in **Vehicular Edge Computing (VEC)** systems.

The goal is to determine:

* Where tasks should execute (local vehicle or VEC server)
* Which services should be cached on servers
* Whether content should be fetched from vehicles or servers

The objective is to **minimize latency-based failure and system cost together**.

---

## Objective Function

# min(ω⋅Cost+(1−ω)⋅Service Failure Rate)
Where:

* **Cost** = server computation + caching + vehicle content access
* **Failure rate** = tasks missing latency deadline

The weight **ω changes dynamically** to balance cost & delay.

---

## System Model

| Component     | Description                          |
| ------------- | ------------------------------------ |
| Vehicles      | Generate tasks and may hold content  |
| Edge Servers  | Limited computing + caching capacity |
| Tasks         | Require content and compute cycles   |
| Content Types | Different services that tasks need   |

---

## Decision Variables

| Variable | Meaning                                 |
| -------- | --------------------------------------- |
| **P1**   | Task offloading (local or which server) |
| **P2**   | Which server caches which content       |
| **P3**   | Content source (vehicle vs server)      |

All optimized **jointly** inside MBAT.

---

## Algorithm Used: MBAT (Multi-Swarm Bat Algorithm)

Main features:

* Separate swarms for P1, P2, P3 decisions
* Bats update using frequency, loudness, pulse rate
* Local search + guided mutation around global best
* Swarms collaborate by moving toward best solution
* Dynamic weights to ensure good cost-delay balance

Effect:

* Better global search early
* Better fine-tuning later
* Reaches Pareto-balanced solution

---

## Parameters (Typical)

| Parameter     | Value |
| ------------- | ----- |
| Population    | 30    |
| Iterations    | 100   |
| Tasks         | 50    |
| Vehicles      | 100   |
| Servers       | 3     |
| Content Types | 10    |

---

## How to Run

### Install Dependencies

```
pip install numpy matplotlib
```

### Execute Program

```
python mbat_optimizer.py
```

---

## Outputs Generated

<img src=output/1.jpeg>

### Convergence graphs

* System efficiency reduction over iterations
* Cost curve
* Failure rate curve

### Pareto Trade-off Plot

* Shows the balance between cost & failure rate

### Strategy Visualization

* Task offloading distribution (local vs server)
* Content retrieval distribution (vehicle vs server)

### Final Metrics

* Best efficiency value
* Final cost & failure probability

---

## Interpretation

| Scenario               | Meaning                              |
| ---------------------- | ------------------------------------ |
| Low cost, high failure | Mostly local computing, less caching |
| High cost, low failure | More server help + caching usage     |
| Balanced optimum       | Best compromise found by MBAT        |

The algorithm gradually shifts from exploration → exploitation and converges to a **balanced VEC strategy**.


