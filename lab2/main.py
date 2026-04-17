import random

#    a    b    c    d    f    g
matrix_1graph = [
    [0,   3,   INF, INF, 1,   INF], # a
    [3,   0,   8,   INF, INF, 3],   # b
    [INF, 3,   0,   1,   INF, 3],   # c
    [INF, INF, 8,   0,   3,   INF], # d
    [3,   INF, INF, 1,   0,   INF], # f
    [3,   INF, 1,   5,   4,   0]    # g
]

def get_start_path(matrix):
    n = len(matrix)
    nodes = list(range(n))
    while True:
        path = [random.choice(nodes)]
        remaining = set(nodes) - {path[0]}
        while remaining:
            current = path[-1]
            neighbors = [node for node in remaining if matrix[current][node] != INF]
            if not neighbors: 
                break
            next_node = random.choice(neighbors)
            path.append(next_node)
            remaining.remove(next_node)
        if len(path) == n and matrix[path[-1]][path[0]] != INF:
            return path