def MinValue(node, visited):
    visited.append(node.data)
    if len(node.children) == 0:
        return node
    node.value = 100000
    for child in node.children:
        temp = MaxValue(child, visited)
        if temp.value < node.value:
            node.value = temp.value
    return node

def MaxValue(node, visited):
    visited.append(node.data)
    if len(node.children) == 0:
        return node
    node.value = -100000
    for child in node.children:
        temp = MinValue(child, visited)
        if temp.value > node.value:
            node.value = temp.value
    return node

def Minimax_Search(state):
    visited = []
    MaxValue(state, visited)
    return visited

class Tree:
    def __init__(self, data, cost=100000):
        self.data = data 
        self.cost = cost
        self.children = []
        self.parent = None
    def add_child(self, child):
        child.parent = self
        self.children.append(child)
    def get_data(self):
        return self.data
    def get_children(self):
        return self.children
    def get_parent(self):
        return self.parent
    def __lt__(self, other):
        return self.cost < other.cost
        
if __name__ == "__main__":
    A = Tree("A")
    B = Tree("B")
    C = Tree("C")
    D = Tree("D")
    E = Tree("E")
    F = Tree("F")
    G = Tree("G")
    H = Tree("H")
    I = Tree("I")
    J = Tree("J")
    K = Tree("K")
    L = Tree("L")
    M = Tree("M")
    N = Tree("N")
    Z = Tree("Z")
    A.add_child(B)
    A.add_child(C)
    B.add_child(D)
    B.add_child(E)
    C.add_child(F)
    C.add_child(G)
    D.add_child(H)
    D.add_child(I)
    E.add_child(J)
    E.add_child(K)
    F.add_child(M)
    F.add_child(N)
    G.add_child(L)
    G.add_child(Z)
    H.value = 2
    I.value = 9
    J.value = 7
    K.value = 4
    M.value = 8
    N.value = 9
    L.value = 3
    Z.value = 5
    visited_nodes = Minimax_Search(A)
    print("Visited order:", visited_nodes)
    print("Root value:", A.value)