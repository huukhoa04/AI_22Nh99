import sys
import csv
import networkx as nx
import matplotlib.pyplot as plt

class Cell:
    def __init__(self, chi_phi, flag=0):
        self.chi_phi = chi_phi
        self.flag = flag

class ToaDo:
    def __init__(self, m, n):
        self.m = m
        self.n = n

class DuongHam:
    def __init__(self, file_name):
        self.m_MaTrix = []
        self.m_TrungTam = None
        self.load_file(file_name)

    def load_file(self, file_name):
              with open(file_name, 'r') as file:
                    reader = csv.reader(file)
                    n = int(next(reader)[0])
                    self.m_MaTrix = [[None] * n for _ in range(n)]
                    m = int(next(reader)[0])
                    n = int(next(reader)[0])
                    self.m_TrungTam = ToaDo(m, n)
                    print(self.m_TrungTam.m, self.m_TrungTam.n)
                    for i, line in enumerate(reader):
                        values = list(map(int, line))
                        for j, chi_phi in enumerate(values):
                            self.m_MaTrix[i][j] = Cell(chi_phi)
                    print(self.m_MaTrix[i][j].chi_phi, end=" ")


    def tim_duong(self):
        arrToaDo = []
        self.m_MaTrix[self.m_TrungTam.m][self.m_TrungTam.n].flag = 1
        arrToaDo.append(self.m_TrungTam)

        arrT = [ToaDo(-1, 0), ToaDo(0, -1), ToaDo(0, 1), ToaDo(1, 0)]

        while arrToaDo:
            t = arrToaDo.pop(0)

            if t.m == 0 or t.n == 0:
                return True

            for ds in arrT:
                Tam = ToaDo(t.m + ds.m, t.n + ds.n)

                if 0 <= Tam.m < len(self.m_MaTrix) and 0 <= Tam.n < len(self.m_MaTrix):
                    if self.m_MaTrix[Tam.m][Tam.n].chi_phi != 0 and self.m_MaTrix[Tam.m][Tam.n].flag == 0:
                        if Tam.m == 0 or Tam.n == 0 or Tam.m == len(self.m_MaTrix) - 1 or Tam.n == len(self.m_MaTrix) - 1:
                            return True
                        arrToaDo.append(Tam)
                        self.m_MaTrix[Tam.m][Tam.n].flag = 1

        return False

    def draw_graph(self):
        G = nx.grid_2d_graph(len(self.m_MaTrix), len(self.m_MaTrix), create_using=nx.DiGraph)
        pos = {(x, y): (y, -x) for x, y in G.nodes()}

        for i in range(len(self.m_MaTrix)):
            for j in range(len(self.m_MaTrix)):
                if self.m_MaTrix[i][j].chi_phi == 0:
                    G.remove_node((i, j))

        nx.draw(G, pos, with_labels=True, node_size=700, node_color="lightblue", font_size=10, font_color="black")
        plt.show()

if __name__ == "__main__":
        dh = DuongHam('data.csv')
        dh.draw_graph()