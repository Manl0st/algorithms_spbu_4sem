import math
import random
from matrices import BERLIN52, WORLD666, MATRIX_1GRAPH

INF = float("inf")
GRAPH_CHOICE = 2
MODE = "classic"
T_START = 100.0
T_MIN = 0.001
ALPHA = 0.95
ITERS_PER_T = 200
MAX_STEPS = 2000

def route_cost(matrix, path):
    total = 0
    for i in range(len(path)):
        weight = matrix[path[i]][path[(i + 1) % len(path)]]
        if weight == INF:
            return INF
        total += weight
    return total

def load_graph(choice):
    if choice == 0:
        return MATRIX_1GRAPH
    elif choice == 1:
        return BERLIN52
    elif choice == 2:
        return WORLD666
    else:
        return MATRIX_1GRAPH
    
def random_path(matrix):
    base_path = list(range(len(matrix)))
    while True:
        path = base_path[:]
        random.shuffle(path)
        if route_cost(matrix, path) != INF:
            return path
        
def mutate_path(path):
    new_path = path[:]
    i, j = random.sample(range(len(new_path)), 2)
    new_path[i], new_path[j] = new_path[j], new_path[i]
    return new_path

def next_temperature(mode, t_now, step, t_start, alpha):
    if mode == "classic":
        return t_now * alpha
    if mode == "cauchy":
        return t_start / (1 + step)
    
def simulated_annealing(matrix, mode, t_start, t_min, alpha, iters_per_t, max_steps):
    t = t_start
    current = random_path(matrix)
    current_cost = route_cost(matrix, current)
    best_path = current[:]
    best_cost = current_cost
    step = 0
    history = [best_cost]
    while t > t_min and step < max_steps:
        for _ in range(iters_per_t):
            candidate_path = mutate_path(current)
            candidate_cost = route_cost(matrix, candidate_path)
            if candidate_cost == INF:
                continue
            delta = candidate_cost - current_cost
            if delta <= 0 or random.random() < math.exp(-delta / t):
                current = candidate_path
                current_cost = candidate_cost
                if current_cost < best_cost:
                    best_path = current[:]
                    best_cost = current_cost
        history.append(best_cost)
        t = next_temperature(mode, t, step, t_start, alpha)
        step += 1
    return {
        "mode": mode,
        "best_path": best_path,
        "best_cost": best_cost,
        "steps": step,
        "history": history,
    }

def main():
    matrix = load_graph(GRAPH_CHOICE)
    result = simulated_annealing(matrix, MODE, T_START, T_MIN, ALPHA, ITERS_PER_T, MAX_STEPS)
    print("Режим:", result["mode"])
    print("Размер графа:", len(matrix))
    print("Лучший путь:", result["best_path"][:10], "..." if len(result["best_path"]) > 10 else "")
    print("Длина лучшего пути:", result["best_cost"])
    print("Шагов охлаждения:", result["steps"])

if __name__ == "__main__":
    main()