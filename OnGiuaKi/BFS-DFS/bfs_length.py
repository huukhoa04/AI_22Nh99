from collections import deque

def bfs_with_length_and_path(graph, start, goal):
    queue = deque([(start, 0, [start])])  # (node, distance, path)
    visited = set()

    while queue:
        node, distance, path = queue.popleft()

        if node == goal:
            return distance, path

        if node not in visited:
            visited.add(node)
            for neighbor in graph[node]:
                if neighbor not in visited:
                    queue.append((neighbor, distance + 1, path + [neighbor]))

    return -1, []  # If the goal is not reachable

# Example usage:
graph = {
    'A': ['B', 'C'],
    'B': ['A', 'D', 'E'],
    'C': ['A', 'F'],
    'D': ['B'],
    'E': ['B', 'F'],
    'F': ['C', 'E']
}
print(bfs_with_length_and_path(graph, 'A', 'F'))  # Output: (2, ['A', 'C', 'F'])