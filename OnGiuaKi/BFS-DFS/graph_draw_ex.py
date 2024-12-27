import matplotlib.pyplot as plt
import networkx as nx
from collections import deque

# Create a graph object
G = nx.Graph()

# Add V
#tập đỉnh
V = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'S']
G.add_nodes_from(V)

# tập cạnh
# Add edges
E = [
    ('A', 'B'), 
    ('A', 'D'), 
    ('A', 'S'),

    ('B', 'C'), 
    ('B', 'F'), 
    ('B', 'G'), 
    ('B', 'S'),
    ('B', 'D'),

    ('C', 'S'),
    ('C', 'F'), 
    
    ('D', 'E'),

    ('E', 'G'),
    ('E', 'F'),

    ('F', 'H'), 
    ('G', 'H')
]
G.add_edges_from(E)

def dfs(graph, start, goal):
    stack = [(start, [start])]
    visited = set()
    all_paths = []

    while stack:
        (vertex, path) = stack.pop()
        if vertex in visited:
            continue

        visited.add(vertex)
        all_paths.append(path)

        for neighbor in graph.neighbors(vertex):
            if neighbor == goal:
                all_paths.append(path + [neighbor])
                print("Visited paths:", all_paths)
                return path + [neighbor]
            else:
                stack.append((neighbor, path + [neighbor]))
    if(all_paths):
        print("Visited paths:", all_paths)
    else:
        print("404 Not Found!")
    return None

# Find path from 'S' to 'G'
path = dfs(G, 'G', 'S')
print("Path from G to S:", path)

# Draw the graph
plt.figure(figsize=(8, 6))
pos = nx.spring_layout(G, seed=10)  # Adjust layout as needed
nx.draw_networkx_nodes(G, pos, node_size=500, node_color='lightblue')
nx.draw_networkx_edges(G, pos, width=2)
nx.draw_networkx_labels(G, pos, font_size=12, font_family='sans-serif')

# Highlight the path
# if path:
#     edge_path = list(zip(path, path[1:]))
#     nx.draw_networkx_edges(G, pos, edgelist=edge_path, width=2, edge_color='r')

plt.title('Graph Visualization')
plt.axis('off')
plt.show()