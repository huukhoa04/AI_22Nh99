import csv
from heapq import heappush, heappop
import numpy as np

def manhattan_distance(current, goal):
    """
    Calculate Manhattan distance to the nearest edge of the castle
    This is our h(x) heuristic function
    """
    x, y = current
    n = goal  # goal is the size of the maze (to reach any edge)
    # Distance to nearest edge (top, bottom, left, or right)
    return min(x, y, n - 1 - x, n - 1 - y)

def is_edge(pos, n):
    """Check if position is at the edge of the castle"""
    x, y = pos
    return x == 0 or x == n - 1 or y == 0 or y == n - 1

def get_neighbors(pos, maze, n):
    """Get valid neighboring positions"""
    x, y = pos
    directions = [(0, 1), (1, 0), (0, -1), (-1, 0)]  # right, down, left, up
    neighbors = []
    for dx, dy in directions:
        new_x, new_y = x + dx, y + dy
        if (0 <= new_x < n and 0 <= new_y < n and 
            maze[new_x][new_y] == 1):  # Check if cell has a tunnel
            neighbors.append((new_x, new_y))
    return neighbors

def astar_escape(maze, start, n):
    """
    A* algorithm implementation for castle escape
    f(x) = g(x) + h(x) where:
    - g(x) is the actual cost from start to current position
    - h(x) is the Manhattan distance to nearest edge
    """
    frontier = []
    heappush(frontier, (0, start))  # (f(x), position)
    came_from = {start: None}
    g_score = {start: 0}
    
    while frontier:
        current = heappop(frontier)[1]
        
        # Check if we've reached an edge
        if is_edge(current, n):
            # Reconstruct path
            path = []
            while current is not None:
                path.append(current)
                current = came_from[current]
            return path[::-1]  # Reverse path to get start-to-goal order
            
        for next_pos in get_neighbors(current, maze, n):
            # g(x) is just the number of steps taken
            tentative_g_score = g_score[current] + 1
            
            if next_pos not in g_score or tentative_g_score < g_score[next_pos]:
                came_from[next_pos] = current
                g_score[next_pos] = tentative_g_score
                # f(x) = g(x) + h(x)
                f_score = tentative_g_score + manhattan_distance(next_pos, n)
                heappush(frontier, (f_score, next_pos))
    
    return None  # No escape path found

def read_input(filename):
    """Read maze from CSV file"""
    with open(filename, 'r') as file:
        reader = csv.reader(file)
        # Read first line: n, D, C
        n, start_row, start_col = map(int, next(reader))
        
        # Read maze
        maze = []
        for _ in range(n):
            row = list(map(int, next(reader)))
            maze.append(row)
            
    return np.array(maze), n, (start_row, start_col)

def write_output(filename, path):
    """Write solution to CSV file"""
    with open(filename, 'w', newline='') as file:
        writer = csv.writer(file)
        if path is None:
            writer.writerow([-1])  # No solution found
        else:
            writer.writerow([len(path)])  # Number of cells in path
            for row, col in path:
                writer.writerow([row, col])  # Write coordinates

def solve_castle_escape(input_file, output_file):
    # Read input
    maze, n, start = read_input(input_file)
    
    # Check if start position has a tunnel
    if maze[start[0]][start[1]] != 1:
        path = None
    else:
        # Find escape path using A*
        path = astar_escape(maze, start, n)
    
    # Write output
    write_output(output_file, path)
    return path

# Run the solution
if __name__ == "__main__":
    input_file = "A_in.csv"
    output_file = "A_out.csv"
    path = solve_castle_escape(input_file, output_file)
    
    if path is None:
        print("Không tìm thấy đường thoát hiểm")
    else:
        print(f"Tìm thấy đường thoát với {len(path)} bước")
        print("Đường đi:", path)