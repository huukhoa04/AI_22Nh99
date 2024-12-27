graph = {
  'A': ['B', 'C'],
  'B': ['D', 'E'],
  'C': ['F', 'G'],
  'D': ['H', 'I'],
  'E': ['J', 'K'],
  'F': ['L', 'M'],
  'G': ['N', 'O'],
  'H': [],
  'I': [],
  'J': [],
  'K': [],
  'L': [],
  'M': [],
  'N': [],
  'O': []
}

def BFS(initialState, goal):
  frontier = [initialState]
  explored = []
  while frontier:
    state = frontier.pop(0)
    explored.append(state)
    if goal == state:
      return explored
    for neighbor in graph[state]:
      if neighbor not in (explored and frontier):
        frontier.append(neighbor)
  return False

def print_edges(graph):
  for node in graph:
    for neighbor in graph[node]:
      print(f"{node} -> {neighbor}")

result = BFS('A', 'K')
if result:
  print('explored:', end=' ')
  for i in result:
    print(i, end=' ')
else:
  print("404 Not Found!")

print("\nAll edges in the graph:")
print_edges(graph)
