class Tree:
    def __init__(self):
        self.nodes = []
        self.edges = []

    def add_nodes(self, data):
        for node in data:
            self.add_node(node)

    def add_node(self, node):
        self.nodes.append(node)

    def get_index(self, node):
        for i, n in enumerate(self.nodes):
            if n.get_label() == node.get_label():
                return i
        return -1

    def add_edges(self, tuple_edges):
        for t in tuple_edges:
            start_label = t[0]
            end_label = t[1]
            w = t[2]
            index_start_label = self.get_index(Node(start_label, None))
            index_end_label = self.get_index(Node(end_label, None))
            self.nodes[index_start_label].child.append(self.nodes[index_end_label])
            self.nodes[index_end_label].parent.append(self.nodes[index_start_label])
            self.edges.append((self.nodes[index_start_label], self.nodes[index_end_label], t[2]))

    def show_nodes(self):
        return [node.get_data() for node in self.nodes]

    def get_edge(self, start_node, end_node):
        try:
            return [edge for edge in self.edges if edge[0] == start_node and edge[1] == end_node][0]
        except:
            return None