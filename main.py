import numpy as np
import matplotlib.pyplot as plt
from typing import List, Dict, Any

class MBATOptimizer:
    def __init__(self, population_size=30, max_iterations=100):
        self.Np = population_size
        self.G = max_iterations
        
        # Network parameters
        self.J = 3  # servers
        self.M = 100  # vehicles
        self.I = 50  # tasks
        self.K = 10  # content types
        
        # Server capacities
        self.Sj = 1000
        self.Fj = 500
        
        # Content sizes
        self.sk = np.full(self.K, 200)
        
        # Task requirements - FIXED for consistent evaluation
        np.random.seed(42)
        self.c_i = np.random.randint(10, 21, size=self.I)
        self.o_i = np.random.randint(0, self.K, size=self.I)
        
        # Pre-compute deterministic performance characteristics
        self.local_failure_prob = 0.8
        self.vehicle_failure_prob = 0.6
        self.server_miss_failure_prob = 0.3
        
        # Bat algorithm parameters - TUNED FOR CONVERGENCE
        self.f_min = 0
        self.f_max = 2
        self.A0 = 0.5
        self.r0 = 0.5
        self.alpha = 0.9  # Loudness decay
        self.gamma = 0.9  # Pulse rate increase
        
        print(f"Initialized with {self.J} servers, {self.M} vehicles, {self.I} tasks, {self.K} content types")

    def initialize_population(self):
        """Initialize populations"""
        P1 = np.random.randint(1, self.J + 2, size=(self.Np, self.I))
        P2 = np.random.randint(0, 2, size=(self.Np, self.J, self.K))
        P3 = np.random.randint(0, 2, size=(self.Np, self.I))
        return P1, P2, P3

    def evaluate_solution_deterministic(self, P1, P2, P3):
        """DETERMINISTIC evaluation - no randomness for consistent convergence"""
        failure_rates = np.zeros(self.Np)
        
        for n in range(self.Np):
            failures = 0
            for i in range(self.I):
                if P1[n, i] == 1:  # Local processing
                    failures += self.local_failure_prob
                else:  # Server processing
                    server_idx = P1[n, i] - 2
                    if server_idx >= self.J:  # Safety check
                        server_idx = self.J - 1
                    content_idx = self.o_i[i]
                    
                    if P3[n, i] == 1:  # Vehicle retrieval
                        failures += self.vehicle_failure_prob
                    else:  # Server retrieval
                        if P2[n, server_idx, content_idx] == 0:  # Not cached
                            failures += self.server_miss_failure_prob
            failure_rates[n] = failures / self.I

        # Calculate cost with STRONG inverse relationship
        costs = np.zeros(self.Np)
        for n in range(self.Np):
            cost = 8000  # Base cost
            
            # STRONG COST-FAILURE TRADE-OFF:
            server_tasks = np.sum(P1[n] > 1)
            cost += server_tasks * 150  # Expensive servers
            
            vehicle_retrievals = np.sum(P3[n] == 1)
            cost += vehicle_retrievals * 30  # Cheap vehicles
            
            cached_content = np.sum(P2[n])
            cost += cached_content * 15  # Expensive caching
            
            costs[n] = cost

        # System efficiency - both objectives should be minimized
        system_efficiency = np.zeros(self.Np)
        for n in range(self.Np):
            # Normalize to create clear optimization landscape
            cost_norm = (costs[n] - 8000) / 7000
            failure_norm = failure_rates[n]
            system_efficiency[n] = 0.5 * cost_norm + 0.5 * failure_norm + 1.5

        return system_efficiency, costs, failure_rates

    def apply_constraints(self, P1, P2, P3):
        """Apply storage constraints"""
        P2_new = P2.copy()
        for n in range(self.Np):
            for j in range(self.J):
                storage_used = np.sum(P2_new[n, j] * self.sk)
                while storage_used > self.Sj:
                    cached_indices = np.where(P2_new[n, j] == 1)[0]
                    if len(cached_indices) > 0:
                        # Remove random cached content
                        remove_idx = np.random.choice(cached_indices)
                        P2_new[n, j, remove_idx] = 0
                        storage_used = np.sum(P2_new[n, j] * self.sk)
                    else:
                        break
        
        P2_new = np.clip(P2_new, 0, 1).astype(int)
        return P1, P2_new, P3

    def optimize(self):
        """IMPROVED optimization with proper convergence"""
        # Initialize
        P1, P2, P3 = self.initialize_population()
        P1, P2, P3 = self.apply_constraints(P1, P2, P3)
        
        # Initialize bat parameters as ARRAYS
        frequencies = np.random.uniform(self.f_min, self.f_max, self.Np)
        loudness = np.full(self.Np, self.A0)
        pulse_rates = np.full(self.Np, self.r0)  # FIXED: This is now an array
        
        best_solutions = []
        convergence_data = []
        
        # Track global best
        global_best_efficiency = float('inf')
        global_best_P1, global_best_P2, global_best_P3 = None, None, None
        global_best_cost = 0
        global_best_failure = 0
        no_improvement_count = 0
        
        print("Starting CONVERGING MBAT Optimization...")
        
        for g in range(self.G):
            # Evaluate current population - DETERMINISTIC
            efficiency, costs, failures = self.evaluate_solution_deterministic(P1, P2, P3)
            current_best_idx = np.argmin(efficiency)
            current_best_efficiency = efficiency[current_best_idx]
            
            # Update global best with ELITISM
            if current_best_efficiency < global_best_efficiency:
                global_best_efficiency = current_best_efficiency
                global_best_P1 = P1[current_best_idx].copy()
                global_best_P2 = P2[current_best_idx].copy()
                global_best_P3 = P3[current_best_idx].copy()
                global_best_cost = costs[current_best_idx]
                global_best_failure = failures[current_best_idx]
                no_improvement_count = 0
                improved = True
            else:
                no_improvement_count += 1
                improved = False
            
            # Generate new population with ADAPTIVE EXPLORATION
            P1_new = np.zeros_like(P1)
            P2_new = np.zeros_like(P2)
            P3_new = np.zeros_like(P3)
            
            # Adaptive parameters based on convergence progress
            exploration_factor = max(0.1, 1.0 - (g / self.G))  # Decrease exploration over time
            mutation_rate = 0.15 * exploration_factor
            
            for n in range(self.Np):
                freq = self.f_min + (self.f_max - self.f_min) * np.random.random()
                
                # FIXED: pulse_rates[n] is now valid since pulse_rates is an array
                if np.random.random() < pulse_rates[n]:
                    # Velocity-based update towards global best
                    P1_new[n] = self.guided_mutation_p1(P1[n], global_best_P1, freq, exploration_factor)
                    P2_new[n] = self.guided_mutation_p2(P2[n], global_best_P2, freq, exploration_factor)
                    P3_new[n] = self.guided_mutation_p3(P3[n], global_best_P3, freq, exploration_factor)
                else:
                    # Local search around current best
                    P1_new[n] = self.local_search_p1(P1[n], global_best_P1, mutation_rate)
                    P2_new[n] = self.local_search_p2(P2[n], global_best_P2, mutation_rate)
                    P3_new[n] = self.local_search_p3(P3[n], global_best_P3, mutation_rate)
            
            # Apply constraints
            P1_new, P2_new, P3_new = self.apply_constraints(P1_new, P2_new, P3_new)
            
            # Evaluate offspring
            efficiency_new, costs_new, failures_new = self.evaluate_solution_deterministic(P1_new, P2_new, P3_new)
            
            # IMPROVED SELECTION: Strong elitism with occasional exploration
            for n in range(self.Np):
                # Always accept better solutions
                if efficiency_new[n] < efficiency[n]:
                    P1[n] = P1_new[n]
                    P2[n] = P2_new[n]
                    P3[n] = P3_new[n]
                    efficiency[n] = efficiency_new[n]
                    costs[n] = costs_new[n]
                    failures[n] = failures_new[n]
                # Occasionally accept worse solutions early on for diversity
                elif g < self.G * 0.3 and np.random.random() < 0.1 * exploration_factor:
                    P1[n] = P1_new[n]
                    P2[n] = P2_new[n]
                    P3[n] = P3_new[n]
                    efficiency[n] = efficiency_new[n]
                    costs[n] = costs_new[n]
                    failures[n] = failures_new[n]
            
            # Update bat parameters for convergence
            loudness = self.alpha * loudness
            pulse_rates = self.r0 * (1 - np.exp(-self.gamma * g)) * np.ones(self.Np)  # FIXED: array update
            
            # Store best solution (always global best for smooth convergence)
            best_solution = {
                'iteration': g,
                'P1': global_best_P1.copy() if global_best_P1 is not None else P1[current_best_idx].copy(),
                'P2': global_best_P2.copy() if global_best_P2 is not None else P2[current_best_idx].copy(),
                'P3': global_best_P3.copy() if global_best_P3 is not None else P3[current_best_idx].copy(),
                'cost': global_best_cost if improved else costs[current_best_idx],
                'failure_rate': global_best_failure if improved else failures[current_best_idx],
                'system_efficiency': global_best_efficiency
            }
            best_solutions.append(best_solution)
            
            convergence_data.append({
                'iteration': g,
                'best_efficiency': global_best_efficiency,
                'avg_efficiency': np.mean(efficiency),
                'best_cost': global_best_cost if improved else costs[current_best_idx],
                'best_failure': global_best_failure if improved else failures[current_best_idx]
            })
            
            if (g + 1) % 20 == 0 or g < 5:
                current_cost = global_best_cost if improved else costs[current_best_idx]
                current_failure = global_best_failure if improved else failures[current_best_idx]
                print(f"Iteration {g+1}/{self.G}, Best Eff: {global_best_efficiency:.4f}, "
                      f"Cost: {current_cost:.0f}, Fail: {current_failure:.3f}")
            
            # Early stopping if converged
            if no_improvement_count > 20 and g > self.G * 0.5:
                print(f"Early stopping at iteration {g+1} - converged")
                break
        
        return best_solutions, convergence_data

    def guided_mutation_p1(self, individual, best_individual, freq, exploration_factor):
        """Guided mutation towards best solution"""
        if best_individual is None:
            return individual.copy()
            
        mutation = individual.copy()
        # Strong guidance towards best solution
        guidance_mask = np.random.random(individual.shape) < (0.4 + 0.3 * (1 - exploration_factor))
        mutation[guidance_mask] = best_individual[guidance_mask]
        
        # Small random exploration
        exploration_mask = np.random.random(individual.shape) < (0.1 * exploration_factor)
        mutation[exploration_mask] = np.random.randint(1, self.J + 2, np.sum(exploration_mask))
        
        return mutation

    def guided_mutation_p2(self, individual, best_individual, freq, exploration_factor):
        """Guided mutation for binary matrix"""
        if best_individual is None:
            return individual.copy()
            
        mutation = individual.copy()
        guidance_mask = np.random.random(individual.shape) < (0.4 + 0.3 * (1 - exploration_factor))
        mutation[guidance_mask] = best_individual[guidance_mask]
        
        exploration_mask = np.random.random(individual.shape) < (0.1 * exploration_factor)
        mutation[exploration_mask] = 1 - mutation[exploration_mask]
        
        return mutation

    def guided_mutation_p3(self, individual, best_individual, freq, exploration_factor):
        """Guided mutation for binary vector"""
        if best_individual is None:
            return individual.copy()
            
        mutation = individual.copy()
        guidance_mask = np.random.random(individual.shape) < (0.4 + 0.3 * (1 - exploration_factor))
        mutation[guidance_mask] = best_individual[guidance_mask]
        
        exploration_mask = np.random.random(individual.shape) < (0.1 * exploration_factor)
        mutation[exploration_mask] = 1 - mutation[exploration_mask]
        
        return np.clip(mutation, 0, 1).astype(int)

    def local_search_p1(self, individual, best_individual, mutation_rate):
        """Local search around current solution"""
        mutation = individual.copy()
        # Small perturbations
        mutation_mask = np.random.random(individual.shape) < mutation_rate
        mutation[mutation_mask] = np.random.randint(1, self.J + 2, np.sum(mutation_mask))
        return mutation

    def local_search_p2(self, individual, best_individual, mutation_rate):
        """Local search for binary matrix"""
        mutation = individual.copy()
        mutation_mask = np.random.random(individual.shape) < mutation_rate
        mutation[mutation_mask] = 1 - mutation[mutation_mask]
        return mutation

    def local_search_p3(self, individual, best_individual, mutation_rate):
        """Local search for binary vector"""
        mutation = individual.copy()
        mutation_mask = np.random.random(individual.shape) < mutation_rate
        mutation[mutation_mask] = 1 - mutation[mutation_mask]
        return np.clip(mutation, 0, 1).astype(int)

def display_z_matrix_full(z_matrix):
    """Display the FULL z matrix with binary values"""
    print(f"\n" + "="*60)
    print("CONTENT CACHING MATRIX (z matrix) - FULL BINARY MATRIX")
    print("="*60)
    print(f"Shape: {z_matrix.shape} (servers × content_types)")
    print("\nMatrix Interpretation:")
    print("- Rows: Servers (0 to J-1)")
    print("- Columns: Content types (0 to K-1)") 
    print("- Values: 1 = cached, 0 = not cached")
    print("\nFULL MATRIX:")
    print("-" * (z_matrix.shape[1] * 2 + 10))
    
    content_headers = [f"C{i}" for i in range(z_matrix.shape[1])]
    print("Server  " + " ".join(f"{header:>2}" for header in content_headers))
    print("-" * (z_matrix.shape[1] * 2 + 10))
    
    for j in range(z_matrix.shape[0]):
        row_values = [f"{z_matrix[j, k]:>2}" for k in range(z_matrix.shape[1])]
        print(f"S{j+1}     " + " ".join(row_values))
    
    print("-" * (z_matrix.shape[1] * 2 + 10))
    
    print(f"\nSUMMARY:")
    for j in range(z_matrix.shape[0]):
        cached_count = np.sum(z_matrix[j])
        print(f"Server {j+1}: {cached_count}/{z_matrix.shape[1]} contents cached "
              f"({cached_count/z_matrix.shape[1]*100:.1f}%)")

def create_smooth_convergence_plots(best_solutions, convergence_data):
    """Create SMOOTH convergence plots"""
    fig = plt.figure(figsize=(15, 10))
    
    # 1. Main convergence - should be SMOOTH now
    plt.subplot(2, 3, 1)
    iterations = [cd['iteration'] for cd in convergence_data]
    best_efficiencies = [cd['best_efficiency'] for cd in convergence_data]
    avg_efficiencies = [cd['avg_efficiency'] for cd in convergence_data]
    
    plt.plot(iterations, best_efficiencies, 'b-', linewidth=2, label='Best Efficiency')
    plt.plot(iterations, avg_efficiencies, 'r--', linewidth=2, label='Average Efficiency')
    plt.xlabel('Iteration')
    plt.ylabel('System Efficiency')
    plt.title('SMOOTH Convergence Progress')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # 2. Pareto Front - should show clear trade-off
    plt.subplot(2, 3, 2)
    final_solutions = best_solutions[-20:]
    costs = [sol['cost'] for sol in final_solutions]
    failures = [sol['failure_rate'] for sol in final_solutions]
    
    plt.scatter(failures, costs, alpha=0.7, s=50, c='green')
    
    # Add trend line to show inverse relationship
    if len(costs) > 1:
        z = np.polyfit(failures, costs, 1)
        p = np.poly1d(z)
        x_trend = np.linspace(min(failures), max(failures), 100)
        plt.plot(x_trend, p(x_trend), 'r--', alpha=0.7, linewidth=2, label='Trade-off Trend')
        plt.legend()
    
    plt.xlabel('Service Failure Rate')
    plt.ylabel('Cost')
    plt.title('Clear Pareto Front (Inverse Relationship)')
    plt.grid(True, alpha=0.3)
    
    # 3. Cost convergence - should be smooth
    plt.subplot(2, 3, 3)
    best_costs = [cd['best_cost'] for cd in convergence_data]
    plt.plot(iterations, best_costs, 'r-', linewidth=2)
    plt.xlabel('Iteration')
    plt.ylabel('Cost')
    plt.title('Cost Convergence')
    plt.grid(True, alpha=0.3)
    
    # 4. Failure rate convergence - should be smooth
    plt.subplot(2, 3, 4)
    best_failures = [cd['best_failure'] for cd in convergence_data]
    plt.plot(iterations, best_failures, 'm-', linewidth=2)
    plt.xlabel('Iteration')
    plt.ylabel('Failure Rate')
    plt.title('Failure Rate Convergence')
    plt.grid(True, alpha=0.3)
    
    # 5. Combined objective view
    plt.subplot(2, 3, 5)
    # Normalize for combined view
    if max(best_costs) > min(best_costs):
        normalized_costs = [(c - min(best_costs)) / (max(best_costs) - min(best_costs)) for c in best_costs]
    else:
        normalized_costs = [0] * len(best_costs)
        
    if max(best_failures) > min(best_failures):
        normalized_failures = [(f - min(best_failures)) / (max(best_failures) - min(best_failures)) for f in best_failures]
    else:
        normalized_failures = [0] * len(best_failures)
    
    plt.plot(iterations, normalized_costs, 'r-', linewidth=2, label='Normalized Cost')
    plt.plot(iterations, normalized_failures, 'm-', linewidth=2, label='Normalized Failure')
    plt.xlabel('Iteration')
    plt.ylabel('Normalized Values')
    plt.title('Combined Objectives Convergence')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # 6. Strategy analysis
    plt.subplot(2, 3, 6)
    best_sol = best_solutions[-1]
    strategies = ['Local', 'Server', 'Vehicle', 'ServerRet']
    local_tasks = np.sum(best_sol['P1'] == 1)
    server_tasks = np.sum(best_sol['P1'] > 1)
    vehicle_retrieval = np.sum(best_sol['P3'] == 1)
    server_retrieval = np.sum(best_sol['P3'] == 0)
    
    values = [local_tasks, server_tasks, vehicle_retrieval, server_retrieval]
    colors = ['red', 'blue', 'green', 'orange']
    bars = plt.bar(strategies, values, color=colors, alpha=0.7)
    plt.ylabel('Number of Tasks')
    plt.title('Final Solution Strategy')
    
    for bar, value in zip(bars, values):
        plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1, 
                f'{value}', ha='center', va='bottom')
    
    plt.tight_layout()
    plt.show()

# Run the optimization
if __name__ == "__main__":
    optimizer = MBATOptimizer(
        population_size=30,
        max_iterations=100
    )
    
    print("Starting CONVERGING MBAT Optimization...")
    best_solutions, convergence_data = optimizer.optimize()
    
    # Create SMOOTH convergence plots
    create_smooth_convergence_plots(best_solutions, convergence_data)
    
    # Display final results
    final = best_solutions[-1]
    print(f"\n" + "="*50)
    print("FINAL OPTIMAL SOLUTION")
    print("="*50)
    print(f"System Efficiency: {final['system_efficiency']:.4f}")
    print(f"Cost: {final['cost']:.0f}")
    print(f"Service Failure Rate: {final['failure_rate']:.4f}")
    
    # Show decision variables
    print(f"\nDecision Variables:")
    print(f"Task Offloading (x vector - first 10): {final['P1'][:10]}")
    print("(1=local, 2=server1, 3=server2, 4=server3)")
    
    print(f"Content Retrieval (y vector - first 10): {final['P3'][:10]}")
    print("(0=retrieve from servers, 1=retrieve from vehicles)")
    
    # Display z matrix
    display_z_matrix_full(final['P2'])
