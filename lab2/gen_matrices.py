INF = float("inf")

def parse_stp(filename):
    n = 0
    edges = {}
    in_graph = False
    
    with open(filename, 'r') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            if line.startswith("Nodes"):
                n = int(line.split()[1])
            if line == "Section Graph":
                in_graph = True
            if line == "End" and in_graph:
                break
            if in_graph and line.startswith("E"):
                parts = line.split()
                if len(parts) >= 4:
                    u, v, w = int(parts[1]) - 1, int(parts[2]) - 1, int(parts[3])
                    if 0 <= u < n and 0 <= v < n:
                        edges[(u, v)] = w
                        edges[(v, u)] = w
    
    matrix = [[INF] * n for _ in range(n)]
    for i in range(n):
        matrix[i][i] = 0
    for (u, v), w in edges.items():
        matrix[u][v] = w
    return matrix


print("Парсинг berlin52.stp...")
berlin52 = parse_stp("berlin52.stp")
print(f"✓ Готово: {len(berlin52)} вершин\n")

print("Парсинг world666.stp...")
world666 = parse_stp("world666.stp")
print(f"✓ Готово: {len(world666)} вершин\n")

print("Создание matrices.py...")
with open("matrices.py", "w") as f:
    f.write("INF = float('inf')\n\n")
    f.write(f"BERLIN52 = {repr(berlin52)}\n\n")
    f.write(f"WORLD666 = {repr(world666)}\n")

print("✓ matrices.py создан!")
