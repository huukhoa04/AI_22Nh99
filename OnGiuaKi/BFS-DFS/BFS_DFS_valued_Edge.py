from collections import deque
from typing import Dict, List, Set, Tuple

class Graph:
    def __init__(self):
        # Initialize adjacency list with weights
        self.graph: Dict[str, List[Tuple[str, int]]] = {}
    
    def add_edge(self, u: str, v: str, weight: int):
        # Add edges with weights to both vertices (undirected graph)
        if u not in self.graph:
            self.graph[u] = []
        if v not in self.graph:
            self.graph[v] = []
        
        self.graph[u].append((v, weight))
        self.graph[v].append((u, weight))

def bfs_with_weights(graph: Graph, start: str) -> Tuple[Dict[str, int], Dict[str, str]]:
    # Initialize distances and parent tracking
    distances = {start: 0}
    parents = {start: None}
    visited = set()
    queue = deque([start])
    visited.add(start)

    while queue:
        current = queue.popleft()
        
        # Process all neighbors
        for neighbor, weight in graph.graph[current]:
            if neighbor not in visited:
                visited.add(neighbor)
                queue.append(neighbor)
                # Calculate distance including edge weight
                distances[neighbor] = distances[current] + weight
                parents[neighbor] = current

    return distances, parents

def dfs_with_weights(graph: Graph, start: str) -> Tuple[Dict[str, int], Dict[str, str]]:
    distances = {start: 0}
    parents = {start: None}
    visited = set()

    def dfs_recursive(vertex: str):
        visited.add(vertex)
        for neighbor, weight in graph.graph[vertex]:
            if neighbor not in visited:
                distances[neighbor] = distances[vertex] + weight
                parents[neighbor] = vertex
                dfs_recursive(neighbor)

    dfs_recursive(start)
    return distances, parents

def find_path_and_weight(graph: Graph, start: str, end: str) -> Tuple[List[str], int]:
    # Get distances and parents using BFS or DFS
    distances, parents = bfs_with_weights(graph, start)
    
    # Check if end vertex is reachable
    if end not in distances:
        return [], -1  # Return empty path and -1 if destination is not reachable
    
    # Reconstruct path
    path = []
    current = end
    while current is not None:
        path.append(current)
        current = parents[current]
    path.reverse()
    
    # Return path and total weight
    return path, distances[end]

# Modify the main section to test the new function:
if __name__ == "__main__":
    # Create a sample tree
    g = Graph()
    g.add_edge('A', 'B', 4)
    g.add_edge('A', 'C', 3)
    g.add_edge('B', 'D', 2)
    g.add_edge('B', 'E', 5)
    g.add_edge('C', 'F', 6)

    # Find path and weight from A to E
    start, end = 'A', 'D'
    path, total_weight = find_path_and_weight(g, start, end)
    
    if total_weight != -1:
        print(f"\nPath from {start} to {end}:")
        print(" -> ".join(path))
        print(f"Total weight: {total_weight}")
    else:
        print(f"\nNo path exists from {start} to {end}")
