import numpy as np
import pandas as pd

# Define the cost matrix, supply and demand
cost_matrix = np.array([
    [16, 20, 12],  # Factory F1 to W1, W2, W3
    [14, 8, 18],   # Factory F2 to W1, W2, W3
    [26, 24, 16]   # Factory F3 to W1, W2, W3
])

supply = np.array([200, 160, 90])
demand = np.array([180, 120, 150])

def northwest_corner(supply, demand):
    """Implement the Northwest Corner starting solution"""
    supply_copy = supply.copy()
    demand_copy = demand.copy()
    m, n = len(supply), len(demand)
    allocation = np.zeros((m, n))
    
    i, j = 0, 0
    while i < m and j < n:
        # Allocate the minimum of supply and demand
        quantity = min(supply_copy[i], demand_copy[j])
        allocation[i, j] = quantity
        
        # Update supply and demand
        supply_copy[i] -= quantity
        demand_copy[j] -= quantity
        
        # Move to next cell
        if supply_copy[i] <= 1e-10:
            i += 1
        if demand_copy[j] <= 1e-10:
            j += 1
    
    return allocation

def compute_stepping_stone_path(allocation, i_enter, j_enter):
    """Find a closed path starting from the entering cell"""
    m, n = allocation.shape
    
    # Find basic cells (with allocation)
    basic_cells = [(i, j) for i in range(m) for j in range(n) if allocation[i, j] > 0]
    path = [(i_enter, j_enter)]
    
    # Function to find next cell in path
    def find_next(current_path, direction):
        last_cell = current_path[-1]
        current_i, current_j = last_cell
        
        if direction == 'row':
            # Find a basic cell in the same row
            for j in range(n):
                if j != current_j and (current_i, j) in basic_cells and (current_i, j) not in current_path:
                    return (current_i, j)
        else:  # 'column'
            # Find a basic cell in the same column
            for i in range(m):
                if i != current_i and (i, current_j) in basic_cells and (i, current_j) not in current_path:
                    return (i, current_j)
        return None
    
    # Try to construct a path
    current_direction = 'row'  # Start by finding a cell in the same row
    
    # Try all possible basic cells as the second cell in the path
    for second_direction in ['row', 'column']:
        current_direction = second_direction
        next_cell = find_next(path, current_direction)
        if next_cell:
            path.append(next_cell)
            
            # Continue constructing the path
            while len(path) < 100:  # Prevent infinite loops
                current_direction = 'row' if current_direction == 'column' else 'column'
                next_cell = find_next(path, current_direction)
                
                if next_cell:
                    path.append(next_cell)
                    
                    # Check if we can close the loop back to the entering cell
                    if (current_direction == 'row' and next_cell[0] == i_enter) or \
                       (current_direction == 'column' and next_cell[1] == j_enter):
                        # We might be able to close the loop
                        if (current_direction == 'row' and find_next([(next_cell[0], j_enter)], 'column') is not None) or \
                           (current_direction == 'column' and find_next([(i_enter, next_cell[1])], 'row') is not None):
                            path.append((i_enter, j_enter))
                            return path
                else:
                    # Can't continue this path
                    break
            
            # If we couldn't complete the path, reset and try another direction
            path = [(i_enter, j_enter)]
    
    return None  # Couldn't find a valid closed path

def solve_modi(cost_matrix, supply, demand):
    """Implement the Modified Distribution Method (MODI)"""
    # Step 1: Find an initial basic feasible solution
    print("Finding initial solution using Northwest Corner method...")
    allocation = northwest_corner(supply, demand)
    print("Initial allocation:")
    print(pd.DataFrame(allocation, 
                      index=[f"F{i+1}" for i in range(len(supply))], 
                      columns=[f"W{j+1}" for j in range(len(demand))]))
    
    m, n = allocation.shape
    total_cost = np.sum(allocation * cost_matrix)
    print(f"Initial cost: {total_cost}")
    
    iteration = 0
    max_iterations = 20  # Safety measure
    
    while iteration < max_iterations:
        iteration += 1
        print(f"\nIteration {iteration}")
        
        # Step 2: Compute the dual variables (u and v)
        u = np.zeros(m)
        v = np.zeros(n)
        u[0] = 0  # Set u[0] to 0 (arbitrary)
        
        # Mark cells with positive allocation
        basic_cells = [(i, j) for i in range(m) for j in range(n) if allocation[i, j] > 0]
        
        # Compute remaining dual variables
        unknown_vars = True
        while unknown_vars:
            unknown_vars = False
            for i, j in basic_cells:
                if np.isnan(u[i]) and not np.isnan(v[j]):
                    u[i] = cost_matrix[i, j] - v[j]
                elif not np.isnan(u[i]) and np.isnan(v[j]):
                    v[j] = cost_matrix[i, j] - u[i]
                elif np.isnan(u[i]) and np.isnan(v[j]):
                    unknown_vars = True
        
        # Step 3: Compute opportunity costs
        opportunity_costs = np.zeros((m, n))
        min_cost = 0
        entering_cell = None
        
        for i in range(m):
            for j in range(n):
                if allocation[i, j] <= 0:
                    opportunity_costs[i, j] = cost_matrix[i, j] - u[i] - v[j]
                    if opportunity_costs[i, j] < min_cost:
                        min_cost = opportunity_costs[i, j]
                        entering_cell = (i, j)
        
        print("Opportunity costs:")
        print(pd.DataFrame(opportunity_costs, 
                         index=[f"F{i+1}" for i in range(m)], 
                         columns=[f"W{j+1}" for j in range(n)]))
        
        # If all opportunity costs are >= 0, we have an optimal solution
        if min_cost >= -1e-10:
            print("\nOptimal solution found!")
            break
        
        print(f"Entering cell: F{entering_cell[0]+1}-W{entering_cell[1]+1}")
        
        # Step 4: Find the closed path (stepping stone path)
        path = None
        
        # Add a temporary small allocation to the entering cell to find the path
        allocation[entering_cell] = 1e-10
        basic_cells.append(entering_cell)
        
        # Use a more robust path-finding approach
        path = compute_stepping_stone_path(allocation, entering_cell[0], entering_cell[1])
        
        # Remove the temporary allocation
        allocation[entering_cell] = 0
        basic_cells.pop()
        
        if path is None:
            print("Could not find a valid path. The solution might be degenerate.")
            break
        
        print(f"Path found: {[(f'F{i+1}-W{j+1}') for i, j in path]}")
        
        # Step 5: Determine the maximum flow change
        theta = float('inf')
        for idx, cell in enumerate(path):
            if idx % 2 == 1:  # Cells where we subtract flow (odd positions)
                theta = min(theta, allocation[cell])
        
        print(f"Maximum flow change: {theta}")
        
        # Step 6: Update the allocation
        for idx, cell in enumerate(path):
            if idx % 2 == 0:  # Even positions (add)
                allocation[cell] += theta
            else:  # Odd positions (subtract)
                allocation[cell] -= theta
        
        # Calculate new total cost
        total_cost = np.sum(allocation * cost_matrix)
        print(f"Updated cost: {total_cost}")
        
        print("Updated allocation:")
        print(pd.DataFrame(allocation, 
                          index=[f"F{i+1}" for i in range(m)], 
                          columns=[f"W{j+1}" for j in range(n)]))
    
    if iteration >= max_iterations:
        print("Maximum iterations reached without finding an optimal solution.")
    
    # Final allocation and cost
    final_cost = np.sum(allocation * cost_matrix)
    print("\nFinal Solution:")
    print(pd.DataFrame(allocation, 
                      index=[f"F{i+1}" for i in range(m)], 
                      columns=[f"W{j+1}" for j in range(n)]))
    print(f"Total minimum cost: {final_cost}")
    
    # Format the shipping plan
    print("\nOptimal Shipping Plan:")
    for i in range(m):
        for j in range(n):
            if allocation[i, j] > 0:
                print(f"Ship {allocation[i, j]} units from Factory {i+1} to Warehouse {j+1} at {cost_matrix[i, j]} BDT per unit")
    
    return allocation, final_cost

# Check if the problem is balanced
total_supply = np.sum(supply)
total_demand = np.sum(demand)
print(f"Total supply: {total_supply}")
print(f"Total demand: {total_demand}")

if total_supply != total_demand:
    print("The problem is unbalanced.")
    if total_supply > total_demand:
        print(f"Adding dummy warehouse with demand {total_supply - total_demand}")
        demand = np.append(demand, total_supply - total_demand)
        cost_matrix = np.hstack((cost_matrix, np.zeros((len(supply), 1))))
    else:
        print(f"Adding dummy factory with supply {total_demand - total_supply}")
        supply = np.append(supply, total_demand - total_supply)
        cost_matrix = np.vstack((cost_matrix, np.zeros((1, len(demand)))))

# Solve the transportation problem
print("\nCost Matrix:")
print(pd.DataFrame(cost_matrix, 
                  index=[f"F{i+1}" for i in range(len(supply))], 
                  columns=[f"W{j+1}" for j in range(len(demand))]))

solve_modi(cost_matrix, supply, demand)