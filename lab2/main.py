import math
import random

INF = float("inf")

#    a    b    c    d    f    g
MATRIX_1GRAPH = [
    [0,   3,   INF, INF, 1,   INF],  # a
    [3,   0,   8,   INF, INF, 3],    # b
    [INF, 3,   0,   1,   INF, 3],    # c
    [INF, INF, 8,   0,   3,   INF],  # d
    [3,   INF, INF, 1,   0,   INF],  # f
    [3,   INF, 1,   5,   4,   0],    # g
]

# Настраиваемые параметры
ANNEALING_MODE = "classic"  # "classic" или "cauchy"
INITIAL_TEMPERATURE = 100.0
MIN_TEMPERATURE = 0.001
COOLING_COEFFICIENT = 0.95
CAUCHY_COEFFICIENT = 1.0
ITERATIONS_PER_TEMPERATURE = 200
MAX_COOLING_STEPS = 2000
RANDOM_SEED = None


def route_cost(matrix, path):
    total = 0
    size = len(path)

    for i in range(size):
        a = path[i]
        b = path[(i + 1) % size]
        weight = matrix[a][b]
        if weight == INF:
            return INF
        total += weight

    return total


def get_random_valid_path(matrix, max_attempts=10000):
    nodes = list(range(len(matrix)))

    for _ in range(max_attempts):
        random.shuffle(nodes)
        path = nodes[:]
        if route_cost(matrix, path) != INF:
            return path

    return None


def mutate_path_swap_two_vertices(path):
    new_path = path[:]
    i, j = random.sample(range(len(new_path)), 2)
    new_path[i], new_path[j] = new_path[j], new_path[i]
    return new_path


def update_temperature(mode, current_temperature, step, initial_temperature,
                       cooling_coefficient, cauchy_coefficient):
    if mode == "classic":
        return current_temperature * cooling_coefficient

    if mode == "cauchy":
        return initial_temperature / (1 + cauchy_coefficient * (step + 1))

    raise ValueError("Неизвестный режим охлаждения")


def simulated_annealing_tsp(matrix, mode, initial_temperature, min_temperature,
                            cooling_coefficient, cauchy_coefficient,
                            iterations_per_temperature, max_cooling_steps):
    current_path = get_random_valid_path(matrix)
    if current_path is None:
        raise ValueError("Для графа не удалось найти начальный гамильтонов цикл")

    current_cost = route_cost(matrix, current_path)
    best_path = current_path[:]
    best_cost = current_cost

    temperature = initial_temperature
    step = 0
    history = [best_cost]

    while temperature > min_temperature and step < max_cooling_steps:
        for _ in range(iterations_per_temperature):
            candidate_path = mutate_path_swap_two_vertices(current_path)
            candidate_cost = route_cost(matrix, candidate_path)

            if candidate_cost == INF:
                continue

            delta = candidate_cost - current_cost
            if delta <= 0 or random.random() < math.exp(-delta / temperature):
                current_path = candidate_path
                current_cost = candidate_cost

                if current_cost < best_cost:
                    best_path = current_path[:]
                    best_cost = current_cost

        history.append(best_cost)
        temperature = update_temperature(
            mode=mode,
            current_temperature=temperature,
            step=step,
            initial_temperature=initial_temperature,
            cooling_coefficient=cooling_coefficient,
            cauchy_coefficient=cauchy_coefficient,
        )
        step += 1

    return {
        "mode": mode,
        "best_path": best_path,
        "best_cost": best_cost,
        "steps": step,
        "history": history,
    }


def run_annealing(matrix, config):
    return simulated_annealing_tsp(
        matrix=matrix,
        mode=config["mode"],
        initial_temperature=config["initial_temperature"],
        min_temperature=config["min_temperature"],
        cooling_coefficient=config["cooling_coefficient"],
        cauchy_coefficient=config["cauchy_coefficient"],
        iterations_per_temperature=config["iterations_per_temperature"],
        max_cooling_steps=config["max_cooling_steps"],
    )


def main():
    if RANDOM_SEED is not None:
        random.seed(RANDOM_SEED)

    config = {
        "mode": ANNEALING_MODE,
        "initial_temperature": INITIAL_TEMPERATURE,
        "min_temperature": MIN_TEMPERATURE,
        "cooling_coefficient": COOLING_COEFFICIENT,
        "cauchy_coefficient": CAUCHY_COEFFICIENT,
        "iterations_per_temperature": ITERATIONS_PER_TEMPERATURE,
        "max_cooling_steps": MAX_COOLING_STEPS,
    }

    result = run_annealing(MATRIX_1GRAPH, config)

    print("Режим:", result["mode"])
    print("Лучший путь:", result["best_path"])
    print("Длина лучшего пути:", result["best_cost"])
    print("Шагов охлаждения:", result["steps"])


if __name__ == "__main__":
    main()
