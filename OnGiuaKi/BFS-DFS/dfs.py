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
def DFS(initialState, goal, limit):
    if limit < 0:
        return False
    count = 0
    
    frontier = [initialState]
    explored = []
    while frontier:
        state = frontier.pop(len(frontier)-1)
        explored.append(state)
        if goal == state:
            return explored
        for neighbor in graph[state]:
            if neighbor not in (explored and frontier):
                frontier.append(neighbor)
    return False

result = DFS('A', 'H', 2)
if result:
  print('explored:', end=' ')
  for i in result:
    print(i, end=' ')
else:
  print("404 Not Found!")