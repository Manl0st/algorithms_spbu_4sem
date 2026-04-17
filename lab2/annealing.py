import math
import random
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend
import matplotlib.pyplot as plt
from matrices import BERLIN52, WORLD666

INF = float("inf")

MATRIX_1GRAPH = [
    [0,   3,   INF, INF, 1,   INF],
    [3,   0,   8,   INF, INF, 3],
    [INF, 3,   0,   1,   INF, 3],
    [INF, INF, 8,   0,   3,   INF],
    [3,   INF, INF, 1,   0,   INF],
    [3,   INF, 1,   5,   4,   0],
]

GRAPHS = {
    0: ("Built-in (6 вершин)", MATRIX_1GRAPH),
    1: ("BERLIN52 (52 города)", BERLIN52),
    2: ("WORLD666 (666 городов)", WORLD666),
}

def route_cost(matrix, path):
    total = 0
    for i in range(len(path)):
        weight = matrix[path[i]][path[(i + 1) % len(path)]]
        if weight == INF:
            return INF
        total += weight
    return total

def load_graph(choice):
    if choice in GRAPHS:
        return GRAPHS[choice][1]
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

def simulated_annealing(matrix, mode, t_start, t_min, alpha, iters_per_t):
    t = t_start
    current = random_path(matrix)
    current_cost = route_cost(matrix, current)
    best_path = current[:]
    best_cost = current_cost
    initial_cost = current_cost  # Запомним начальную стоимость
    step = 0
    history_cost = [best_cost]
    history_temp = [t]
    
    while t > t_min:
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
        history_cost.append(best_cost)
        t = next_temperature(mode, t, step, t_start, alpha)
        history_temp.append(t)
        step += 1
    
    return {
        "mode": mode,
        "best_path": best_path,
        "best_cost": best_cost,
        "initial_cost": initial_cost,
        "steps": step,
        "history_cost": history_cost,
        "history_temp": history_temp,
    }

def get_graph_choice():
    """Получить выбор графа от пользователя"""
    print("\n=== Выбор графа ===")
    for key, (name, _) in GRAPHS.items():
        print(f"  {key}: {name}")
    
    while True:
        try:
            choice = int(input("Введите номер графа (0-2): "))
            if choice in GRAPHS:
                return choice, len(GRAPHS[choice][1])
            else:
                print("Неверный выбор. Попробуйте снова.")
        except ValueError:
            print("Введите число.")

def get_mode():
    """Получить выбор режима охлаждения"""
    print("\n=== Выбор режима охлаждения ===")
    print("  0: Classic (T = T * ALPHA)")
    print("  1: Cauchy (T = T_START / (1 + step))")
    
    while True:
        try:
            choice = int(input("Введите номер режима (0 или 1): "))
            if choice == 0:
                return "classic"
            elif choice == 1:
                return "cauchy"
            else:
                print("Неверный выбор.")
        except ValueError:
            print("Введите число.")

def get_parameters(mode):
    """Получить параметры алгоритма"""
    print("\n=== Параметры алгоритма ===")
    
    while True:
        try:
            t_start = float(input("T_START (начальная температура, например 100.0): "))
            if t_start > 0:
                break
            print("T_START должна быть положительной.")
        except ValueError:
            print("Введите число.")
    
    while True:
        try:
            t_min = float(input("T_MIN (минимальная температура, например 0.089): "))
            if 0 < t_min < t_start:
                break
            print(f"T_MIN должна быть между 0 и {t_start}.")
        except ValueError:
            print("Введите число.")
    
    if mode == "classic":
        while True:
            try:
                alpha = float(input("ALPHA (коэффициент охлаждения, 0.9-0.99): "))
                if 0 < alpha < 1:
                    break
                print("ALPHA должна быть между 0 и 1.")
            except ValueError:
                print("Введите число.")
    else:
        alpha = None  # Не используется для режима Cauchy
    
    while True:
        try:
            iters_per_t = int(input("ITERS_PER_T (итераций при каждой температуре, например 30): "))
            if iters_per_t > 0:
                break
            print("ITERS_PER_T должна быть положительной.")
        except ValueError:
            print("Введите число.")
    
    return t_start, t_min, alpha, iters_per_t

def print_results(result, graph_name, graph_size):
    """Вывести результаты"""
    print("\n" + "="*60)
    print("РЕЗУЛЬТАТЫ РАБОТЫ АЛГОРИТМА")
    print("="*60)
    
    print("\n[ПАРАМЕТРЫ]")
    print(f"  График: {graph_name} ({graph_size} вершин)")
    print(f"  Режим охлаждения: {result['mode'].upper()}")
    print(f"  T_START: {result.get('t_start', 'N/A')}")
    print(f"  T_MIN: {result.get('t_min', 'N/A')}")
    if result.get('alpha') is not None:
        print(f"  ALPHA: {result['alpha']}")
    print(f"  ITERS_PER_T: {result.get('iters_per_t', 'N/A')}")
    
    print("\n[РЕЗУЛЬТАТЫ]")
    initial = result.get('initial_cost')
    best = result['best_cost']
    if initial is not None and isinstance(initial, (int, float)) and initial > 0:
        improvement = ((initial - best) / initial * 100)
        print(f"  Начальная длина пути: {initial:.2f}")
        print(f"  Лучшая длина пути: {best:.2f}")
        print(f"  Улучшение: {improvement:.1f}%")
    else:
        print(f"  Лучшая длина пути: {best:.2f}")
    print(f"  Шагов охлаждения: {result['steps']}")
    if graph_size <= 10:
        print(f"  Лучший путь: {result['best_path']}")
    else:
        print(f"  Лучший путь (первые 10): {result['best_path'][:10]} ...")
    
    print("\n" + "="*60 + "\n")

def plot_results(result):
    """Отобразить два графика"""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    
    # График 1: Лучшая стоимость vs шаги
    steps = list(range(len(result['history_cost'])))
    ax1.plot(steps, result['history_cost'], linewidth=1.5, color='blue')
    ax1.set_xlabel('Шаг охлаждения', fontsize=11)
    ax1.set_ylabel('Лучшая длина пути', fontsize=11)
    ax1.set_title('Сходимость алгоритма', fontsize=12, fontweight='bold')
    ax1.grid(True, alpha=0.3)
    
    # График 2: Температура vs шаги
    ax2.plot(steps, result['history_temp'], linewidth=1.5, color='red')
    ax2.set_xlabel('Шаг охлаждения', fontsize=11)
    ax2.set_ylabel('Температура', fontsize=11)
    ax2.set_title('Охлаждение температуры', fontsize=12, fontweight='bold')
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    filename = "tsp_results.png"
    plt.savefig(filename, dpi=100)
    print(f"\n📊 Графики сохранены в: {filename}")
    plt.close()

def main():
    print("\n╔═══════════════════════════════════════════╗")
    print("║  SIMULATED ANNEALING - TSP SOLVER        ║")
    print("╚═══════════════════════════════════════════╝")
    
    # Получить параметры
    graph_choice, graph_size = get_graph_choice()
    graph_name = GRAPHS[graph_choice][0]
    mode = get_mode()
    t_start, t_min, alpha, iters_per_t = get_parameters(mode)
    
    # Запустить алгоритм
    print("\n⏳ Выполняется поиск оптимального маршрута...")
    matrix = load_graph(graph_choice)
    result = simulated_annealing(matrix, mode, t_start, t_min, alpha, iters_per_t)
    
    # Сохранить параметры в результат для вывода
    result['t_start'] = t_start
    result['t_min'] = t_min
    result['alpha'] = alpha
    result['iters_per_t'] = iters_per_t
    
    # Вывести результаты
    print_results(result, graph_name, graph_size)
    
    # Показать графики
    plot_results(result)

if __name__ == "__main__":
    main()