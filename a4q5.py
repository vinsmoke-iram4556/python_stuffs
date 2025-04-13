import numpy as np

# Define the transportation problem
costs = np.array([
    [4, 3, 1, 2, 6],  # A to P, Q, R, S, T
    [5, 2, 3, 4, 5],  # B to P, Q, R, S, T
    [3, 5, 6, 3, 2],  # C to P, Q, R, S, T
    [2, 4, 4, 5, 3]   # D to P, Q, R, S, T
])

supply = np.array([80, 60, 40, 20])  # A, B, C, D
demand = np.array([60, 60, 30, 40, 10])  # P, Q, R, S, T

# 1. North-West Corner Rule
def northwest_corner(supply, demand):
    supply_copy = supply.copy()
    demand_copy = demand.copy()
    allocation = np.zeros((len(supply), len(demand)))
    
    i, j = 0, 0
    while i < len(supply) and j < len(demand):
        # Allocate the minimum of available supply and demand
        amount = min(supply_copy[i], demand_copy[j])
        allocation[i, j] = amount
        
        # Update remaining supply and demand
        supply_copy[i] -= amount
        demand_copy[j] -= amount
        
        # Move to next cell
        if supply_copy[i] == 0:
            i += 1
        elif demand_copy[j] == 0:
            j += 1
    
    return allocation

# 2. Least Cost Method
def least_cost(supply, demand, costs):
    supply_copy = supply.copy()
    demand_copy = demand.copy()
    allocation = np.zeros((len(supply), len(demand)))
    
    while sum(supply_copy) > 0 and sum(demand_copy) > 0:
        # Find cell with minimum cost where allocation is possible
        min_cost = float('inf')
        min_i, min_j = 0, 0
        
        for i in range(len(supply_copy)):
            if supply_copy[i] <= 0:
                continue
                
            for j in range(len(demand_copy)):
                if demand_copy[j] <= 0:
                    continue
                    
                if costs[i, j] < min_cost:
                    min_cost = costs[i, j]
                    min_i, min_j = i, j
        
        # Make allocation
        amount = min(supply_copy[min_i], demand_copy[min_j])
        allocation[min_i, min_j] = amount
        
        # Update remaining supply and demand
        supply_copy[min_i] -= amount
        demand_copy[min_j] -= amount
    
    return allocation

# Solve using both methods
nw_solution = northwest_corner(supply, demand)
lc_solution = least_cost(supply, demand, costs)

# Calculate total costs
nw_cost = np.sum(nw_solution * costs)
lc_cost = np.sum(lc_solution * costs)

# Display results in a readable format
sources = ['A', 'B', 'C', 'D']
destinations = ['P', 'Q', 'R', 'S', 'T']

def print_solution(solution, method_name, total_cost):
    print(f"\n{method_name} Solution:")
    print("Basic Feasible Solution:")
    
    for i in range(len(sources)):
        for j in range(len(destinations)):
            if solution[i, j] > 0:
                print(f"x{i+1}{j+1} = {int(solution[i, j])} ({sources[i]} to {destinations[j]})")
    
    print(f"Total Cost: {int(total_cost)}")

# Print solutions
print_solution(nw_solution, "North-West Corner Rule", nw_cost)
print_solution(lc_solution, "Least Cost Method", lc_cost)
    

