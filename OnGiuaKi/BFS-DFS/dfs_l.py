from collections import defaultdict

# This class represents a directed graph using adjacency
# list representation
class Graph:

    def __init__(self, vertices):
        # No. of vertices
        self.V = vertices

        # default dictionary to store graph
        self.graph = defaultdict(list)

    # function to add an edge to graph
    def addEdge(self, u, v):
        self.graph[u].append(v)

    # A function to perform a Depth-Limited search
    # from given source 'src'
    def DLS(self, src, target, maxDepth, visited):

        visited.append(src)
        if src == target:
            return True

        # If reached the maximum depth, stop recursing.
        if maxDepth <= 0:
            return False

        # Recur for all the vertices adjacent to this vertex
        for i in self.graph[src]:
            if self.DLS(i, target, maxDepth - 1, visited):
                return True
        return False

    # IDDFS to search if target is reachable from v.
    # It uses recursive DLS()
    def IDDFS(self, src, target, maxDepth):
        visited = []
        # Repeatedly depth-limit search till the maximum depth
        for i in range(maxDepth):
            if self.DLS(src, target, i, visited):
                print("Search order:", visited)
                return True
            visited.clear()
        print("Search order:", visited)
        return False

# Create a graph given in the above diagram
g = Graph(15)
g.addEdge('A', 'B')
g.addEdge('A', 'C')
g.addEdge('B', 'D')
g.addEdge('B', 'E')
g.addEdge('C', 'F')
g.addEdge('C', 'G')
g.addEdge('D', 'H')
g.addEdge('D', 'I')
g.addEdge('E', 'J')
g.addEdge('E', 'K')
g.addEdge('F', 'L')
g.addEdge('F', 'M')
g.addEdge('G', 'N')
g.addEdge('G', 'O')

target = 'O'
maxDepth = 4
src = 'A'

if g.IDDFS(src, target, maxDepth) == True:
    print("Target is reachable from source within max depth")
else:
    print("Target is NOT reachable from source within max depth")