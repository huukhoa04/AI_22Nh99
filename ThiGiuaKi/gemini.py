import networkx as nx
import csv
import heapq
import matplotlib.pyplot as plt

def read_data(filename):
    """Đọc dữ liệu từ file CSV và tạo đồ thị"""
    G = nx.Graph()
    with open(filename, 'r') as f:
        reader = csv.reader(f)
        rows, cols, start_row = next(reader)
        rows, cols, start_row = int(rows), int(cols), int(start_row) - 1
        start_col = int(cols / 2)

        for i, row in enumerate(reader):
            for j, val in enumerate(row):
                if val == '1':
                    G.add_node((i, j))
                    if i > 0 and row[j-1] == '1':
                        G.add_edge((i, j), (i, j-1))
                    if j > 0 and row[j-1] == '1':
                        G.add_edge((i, j), (i-1, j))

        return G, (start_row, start_col)

def heuristic(a, b):
    """Hàm heuristic tính khoảng cách Manhattan giữa hai điểm a và b"""
    return abs(a[0] - b[0]) + abs(a[1] - b[1])

def a_star_search(graph, start, goal):
    """Thuật toán A* tìm đường đi ngắn nhất từ start đến goal"""
    open_set = []
    heapq.heappush(open_set, (0, start))
    came_from = {}
    g_score = {node: float('inf') for node in graph.nodes}
    g_score[start] = 0
    f_score = {node: float('inf') for node in graph.nodes}
    f_score[start] = heuristic(start, goal)

    while open_set:
        _, current = heapq.heappop(open_set)

        if current == goal:
            return reconstruct_path(came_from, current)

        for neighbor in graph.neighbors(current):
            tentative_g_score = g_score[current] + 1  # Giả sử mỗi cạnh có trọng số bằng 1

            if tentative_g_score < g_score[neighbor]:
                came_from[neighbor] = current
                g_score[neighbor] = tentative_g_score
                f_score[neighbor] = g_score[neighbor] + heuristic(neighbor, goal)
                if neighbor not in [i[1] for i in open_set]:
                    heapq.heappush(open_set, (f_score[neighbor], neighbor))

    return None

def reconstruct_path(came_from, current):
    """Hàm tái tạo đường đi từ start đến goal"""
    total_path = [current]
    while current in came_from:
        current = came_from[current]
        total_path.append(current)
    return total_path[::-1]

def draw_graph(G, path=None):
    """Vẽ đồ thị sử dụng NetworkX và Matplotlib"""
    pos = {node: (node[1], -node[0]) for node in G.nodes}  # Đảo ngược trục y để vẽ đúng hướng
    nx.draw(G, pos, with_labels=True, node_size=300, node_color='lightblue', font_size=8, font_weight='bold')
    
    if path:
        path_edges = list(zip(path, path[1:]))
        nx.draw_networkx_nodes(G, pos, nodelist=path, node_color='red')
        nx.draw_networkx_edges(G, pos, edgelist=path_edges, edge_color='red', width=2)
    
    plt.show()

def main():
    G, start = read_data('data.csv')

    # Tìm tất cả các ô biên
    rows = max(node[0] for node in G.nodes) + 1
    cols = max(node[1] for node in G.nodes) + 1
    goals = [(i, 0) for i in range(rows)] + [(i, cols-1) for i in range(rows)] + \
            [(0, j) for j in range(1, cols-1)] + [(rows-1, j) for j in range(1, cols-1)]

    # Tìm đường đi ngắn nhất đến bất kỳ ô biên nào
    shortest_path = None
    for goal in goals:
        path = a_star_search(G, start, goal)
        if path and (not shortest_path or len(path) < len(shortest_path)):
            shortest_path = path

    if shortest_path is None:
        print(-1)
    else:
        print(len(shortest_path) - 1)  # Số bước đi
        for step in shortest_path:
            print(step)
    
    # Vẽ đồ thị và đường đi ngắn nhất
    draw_graph(G, shortest_path)

if __name__ == "__main__":
    main()