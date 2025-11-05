VaCo: Vehicle-assisted Collaborative Caching System Simulation

🌟 Project Overview

This project implements and simulates VaCo (Vehicle-assisted Collaborative caching system), a novel architecture for Vehicular Edge Computing (VEC) networks, as detailed in the research paper. The core goal of VaCo is to address the limited storage capacity of VEC servers by dynamically utilizing the caching resources of surrounding vehicles to support intelligent service content for task offloading.

The simulation focuses on jointly optimizing the Service Failure Rate and the Total Cost in a highly dynamic VEC environment using a multi-objective optimization approach.

🚀 Key Features

Optimization Algorithm: Utilizes a custom Multi-Swarm Collaborative Bat Algorithm (MBAT), implemented in Python, to efficiently solve the Mixed-Integer Nonlinear Programming (MINLP) problem.

Multi-Objective Optimization: Jointly minimizes two conflicting objectives:

Service Failure Rate (F): The proportion of tasks that exceed the maximum tolerable latency ($t_{max}$).

Total Cost (C): Combines penalty cost for task failure and content access cost for invoking vehicle storage resources.

Deterministic Evaluation Model: The initial phase uses a simplified, deterministic evaluation model with fixed failure probabilities and a linear cost function for rapid development and testing of the MBAT's convergence behavior.

Dynamic Weight Index: Uses a Pareto-based optimization scheme to dynamically adjust the trade-off weight ($\omega$) between cost and service failure rate, ensuring fairness between service providers and vehicles.

🛠️ Technology Stack & Setup

This project uses Python as the primary environment for implementing the core optimization algorithm and generating the initial plots.

Prerequisites

You will need the following software installed:

Python (3.8+): Essential for running the MBAT algorithm (MBATOptimizer class), numerical computation (numpy), and plotting (matplotlib).

MATLAB (Optional): May be required later for implementing and comparing advanced schemes like NSGA-II and EHGSA, or for running a more complex VEC network physical layer model.

Real-world Dataset: The final version will use the Roman taxis mobility trace dataset for realistic scenario generation.

Current Simulation Parameters (MBATOptimizer Default)

The current Python implementation uses the following concrete, simplified parameters for initial convergence testing:

Parameter

Symbol

Value

Description

VEC Servers

$J$

3

Number of roadside edge servers.

Total Tasks

$I$

50

Number of computation tasks generated per time slot.

Vehicles

$M$

100

Number of vehicles in the system (resource providers).

Content Types

$K$

10

Number of unique service contents available.

Server Storage

$S_j$

1000 MB

Storage capacity per VEC server.

Content Size

$s_k$

200 MB

Size of each content type (fixed).

Project Structure

VaCo-VEC-Simulation/
├── README.md
├── src/
│   ├── python/
│   │   ├── mbatt_optimizer.py  # MBATOptimizer class (Core implementation)
│   │   └── data_processor.py   # Script to clean/format raw mobility data
│   ├── matlab/
│   │   ├── Main_Simulation.m   # Main script for running comparisons (Planned)
│   │   └── VaCo_Model.m        # Functions defining the VEC system model (Planned)
├── data/
│   └── raw/
│       └── roma_taxi_traces.csv # Original dataset files
└── results/
    ├── convergence_plots/
    └── performance_metrics.csv


📚 Core Methodology Summary

1. Decision Variables

The MBAT algorithm optimizes three sets of decision variables for each individual in the population:

Task Offloading Decisions ($P_1$ / $x_{i}$): Decides where task $i$ is executed (1 = local, 2 to $J+1$ = Server $j$).

Content Caching Decisions ($P_2$ / $z_{jk}$): Binary matrix indicating if VEC server $j$ caches service content $k$.

Content Retrieval Decisions ($P_3$ / $y_{i}$): Binary vector indicating if task $i$ attempts to retrieve required content from the vehicle cluster (1) or the server cluster (0).

2. Objective Function

The system seeks to minimize the objective function, $\text{PO}$, by balancing the two objectives. The current model uses an equal initial weight, $\omega = 0.5$, for demonstration:

$$ \min_{x_{i}, z_{jk}, y_{i}} 0.5 C + (1 - 0.5) F $$

Where:

$F$ is the Service Failure Rate (Eq. 14 in the paper).

$C$ is the Total Cost (Eq. 17 in the paper).

$\omega$ is the dynamic weight factor optimized by the Pareto-based scheme.

3. Constraints

The algorithm rigorously enforces the Storage Constraint on VEC servers, ensuring the size of cached contents does not exceed the server's storage capacity ($S_j$).

📈 Analysis and Visualization

The Python environment is configured to generate real-time plots of the optimization process using matplotlib:

SMOOTH Convergence Progress: Tracks the Best and Average System Efficiency over iterations, demonstrating the MBAT's search capability.

Clear Pareto Front (Inverse Relationship): Plots Cost vs. Service Failure Rate to visualize the fundamental trade-off of the multi-objective problem.

Combined Objectives Convergence: Shows the normalized convergence of both Cost and Failure Rate, which the optimizer seeks to minimize simultaneously.

Final Solution Strategy: Bar chart quantifying the distribution of tasks among different strategies (Local execution, Server Offloading, Vehicle Retrieval, Server Retrieval) in the final optimal solution.
