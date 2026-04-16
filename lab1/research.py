import math
import random
import statistics
import sys


OPTIMAL_VALUE = -959.6407
EPSILON = 0.001
MAX_ITER_LIMIT = 5000
FIXED_ITERATIONS = 1000
NUM_RUNS = 25
CONVERGENCE_ITERS = 500

SCOPE = (-512.0, 512.0)

POP_SIZES = [50, 100, 200]
MUTATION_RATES = [0.1, 0.2, 0.3]

PSO_X_VALUES = [0.5, 0.729, 1.0]
PSO_F1F2_VALUES = [(1.5, 2.5), (2.05, 2.05), (2.5, 1.5)]
PSO_FIXED_X_FOR_F1F2 = 1.0
PSO_FIXED_N_FOR_F1F2 = 100

PSO_BEST_F1 = 2.5
PSO_BEST_F2 = 1.5

BEST_PSO_X_FOR_COMPARISON = 1.0


def fitness_function(x, y):
    return -(y + 47) * math.sin(math.sqrt(abs(x / 2 + (y + 47)))) - x * math.sin(
        math.sqrt(abs(x - (y + 47)))
    )


def fitness_function_value(chromosome):
    return fitness_function(chromosome[0], chromosome[1])


def selection(population, size=4):
    competition = []
    while len(competition) < size:
        competition.append(random.choice(population))
    best = None
    best_val = None
    for item in competition:
        value = fitness_function_value(item)
        if best is None or value < best_val:
            best = item
            best_val = value
    return best


def mutate(chromosome, rate):
    if random.random() < rate:
        for i in range(len(chromosome)):
            chromosome[i] = random.uniform(*SCOPE)
    return chromosome


def blx_alpha(chrom1, chrom2, alpha=0.5):
    new_chromosomes = [[], []]
    for i in range(2):
        for j in range(2):
            d = abs(chrom1[j] - chrom2[j])
            new_val = random.uniform(
                min(chrom1[j], chrom2[j]) - alpha * d,
                max(chrom1[j], chrom2[j]) + alpha * d,
            )
            new_val = max(SCOPE[0], min(SCOPE[1], new_val))
            new_chromosomes[i].append(new_val)
    return new_chromosomes


def discrete_recombination(chrom1, chrom2):
    child1 = [random.choice([chrom1[0], chrom2[0]]), random.choice([chrom1[1], chrom2[1]])]
    child2 = [random.choice([chrom1[0], chrom2[0]]), random.choice([chrom1[1], chrom2[1]])]
    return [child1, child2]


def mean_std(values):
    if not values:
        return 0.0, 0.0
    if len(values) == 1:
        return float(values[0]), 0.0
    return float(statistics.mean(values)), float(statistics.pstdev(values))


def avg_history(histories):
    if not histories:
        return []
    min_len = min(len(h) for h in histories)
    if min_len <= 0:
        return []
    result = []
    for i in range(min_len):
        col = []
        for h in histories:
            col.append(h[i])
        result.append(float(statistics.mean(col)))
    return result


def run_ga(crossover_type, mut_rate, pop_size, max_iter=None, target_epsilon=None):
    best_chromosome = None
    best_fitness_value = None
    iter_count = 0
    pop = [[random.uniform(*SCOPE) for _ in range(2)] for _ in range(pop_size)]
    best_fitness_history = []

    while True:
        new_pop = pop[:]
        while len(new_pop) < pop_size * 2:
            chrom1 = selection(pop)
            chrom2 = selection(pop)
            if crossover_type == "BLX-alpha":
                chrom3, chrom4 = blx_alpha(chrom1, chrom2)
            else:
                chrom3, chrom4 = discrete_recombination(chrom1, chrom2)
            mutate(chrom3, mut_rate)
            mutate(chrom4, mut_rate)
            new_pop.append(chrom3)
            new_pop.append(chrom4)

        pop = sorted(new_pop, key=fitness_function_value)[:pop_size]
        fit_val = fitness_function_value(pop[0])
        if best_chromosome is None or best_fitness_value > fit_val:
            best_fitness_value = fit_val
            best_chromosome = pop[0]
        best_fitness_history.append(best_fitness_value)
        iter_count += 1

        if target_epsilon is not None and abs(best_fitness_value - OPTIMAL_VALUE) < target_epsilon:
            break
        if max_iter is not None and iter_count >= max_iter:
            break
        if iter_count >= MAX_ITER_LIMIT:
            break

    return best_chromosome, best_fitness_value, iter_count, best_fitness_history


def run_pso(f1, f2, X, particle_size, max_iter=None, target_epsilon=None):
    gbest = [random.uniform(*SCOPE), random.uniform(*SCOPE)]
    fitness_gbest = fitness_function(gbest[0], gbest[1])
    particle_list = []

    class Particle:
        def __init__(self, x, y, v0):
            self.fitness = fitness_function(x, y)
            self.position = [x, y]
            self.pbest = [x, y]
            self.fitness_pbest = self.fitness
            self.v = v0

    def move(p):
        nonlocal gbest, fitness_gbest
        r1 = random.uniform(0, 1)
        r2 = random.uniform(0, 1)
        v_x = X * (p.v[0] + f1 * r1 * (p.pbest[0] - p.position[0]) + f2 * r2 * (gbest[0] - p.position[0]))
        v_y = X * (p.v[1] + f1 * r1 * (p.pbest[1] - p.position[1]) + f2 * r2 * (gbest[1] - p.position[1]))
        p.v = [v_x, v_y]
        p.position[0] = min(SCOPE[1], max(SCOPE[0], p.position[0] + v_x))
        p.position[1] = min(SCOPE[1], max(SCOPE[0], p.position[1] + v_y))
        p.fitness = fitness_function(p.position[0], p.position[1])
        if p.fitness < p.fitness_pbest:
            p.fitness_pbest = p.fitness
            p.pbest = p.position[:]
        if p.fitness < fitness_gbest:
            fitness_gbest = p.fitness
            gbest = p.position[:]

    for _ in range(particle_size):
        v = [random.uniform(*SCOPE), random.uniform(*SCOPE)]
        particle_list.append(Particle(random.uniform(*SCOPE), random.uniform(*SCOPE), v))

    best_history = []
    iter_count = 0

    while True:
        for p in particle_list:
            move(p)
        best_history.append(fitness_gbest)
        iter_count += 1

        if target_epsilon is not None and abs(fitness_gbest - OPTIMAL_VALUE) < target_epsilon:
            break
        if max_iter is not None and iter_count >= max_iter:
            break
        if iter_count >= MAX_ITER_LIMIT:
            break

    return gbest, fitness_gbest, iter_count, best_history


def run_experiment_1_ga(verbose=True):
    results = {"BLX-alpha": {}, "Discrete": {}}
    for crossover in ["BLX-alpha", "Discrete"]:
        for pop_size in POP_SIZES:
            for mut_rate in MUTATION_RATES:
                config_name = "pop={} mut={}".format(pop_size, mut_rate)
                iterations = []
                for _ in range(NUM_RUNS):
                    _, _, iter_count, _ = run_ga(crossover, mut_rate, pop_size, target_epsilon=EPSILON)
                    iterations.append(iter_count)
                avg, std = mean_std(iterations)
                results[crossover][config_name] = {"mean": avg, "std": std}

    if verbose:
        print("")
        print("=" * 60)
        print("ЭКСПЕРИМЕНТ 1 (GA): сколько итераций нужно для точности {}".format(EPSILON))
        print("=" * 60)
        print("")
        for crossover in ["BLX-alpha", "Discrete"]:
            print("GA {}:".format(crossover))
            for name in results[crossover]:
                avg = results[crossover][name]["mean"]
                std = results[crossover][name]["std"]
                print("  {} -> среднее {} итер. (разброс {})".format(name, int(avg), int(std)))
            print("")

    return results


def run_experiment_1_pso_x(verbose=True):
    results_by_x = {}
    for X_val in PSO_X_VALUES:
        results_by_x[X_val] = {}
        for n in POP_SIZES:
            config_name = "n={}".format(n)
            iterations = []
            for _ in range(NUM_RUNS):
                _, _, iter_count, _ = run_pso(PSO_BEST_F1, PSO_BEST_F2, X_val, n, target_epsilon=EPSILON)
                iterations.append(iter_count)
            avg, std = mean_std(iterations)
            results_by_x[X_val][config_name] = {"mean": avg, "std": std}

    if verbose:
        print("")
        print("=" * 60)
        print(
            "ЭКСПЕРИМЕНТ 1 (PSO-X,n): фиксируем f1={} f2={}, меняем X и n".format(PSO_BEST_F1, PSO_BEST_F2)
        )
        print("=" * 60)
        for X_val in PSO_X_VALUES:
            print("")
            print("PSO X={}:".format(X_val))
            for name in results_by_x[X_val]:
                avg = results_by_x[X_val][name]["mean"]
                std = results_by_x[X_val][name]["std"]
                print("  {} -> среднее {} итер. (разброс {})".format(name, int(avg), int(std)))

    return results_by_x


def run_experiment_1_pso_f1f2(verbose=True):
    results = {}
    for f1, f2 in PSO_F1F2_VALUES:
        config_name = "f1={} f2={}".format(f1, f2)
        iterations = []
        for _ in range(NUM_RUNS):
            _, _, iter_count, _ = run_pso(
                f1,
                f2,
                PSO_FIXED_X_FOR_F1F2,
                PSO_FIXED_N_FOR_F1F2,
                target_epsilon=EPSILON,
            )
            iterations.append(iter_count)
        avg, std = mean_std(iterations)
        results[config_name] = {"mean": avg, "std": std}

    if verbose:
        print("")
        print("=" * 60)
        print(
            "ЭКСПЕРИМЕНТ 1 (PSO-f1f2): фиксируем X={} и n={}, меняем f1,f2".format(
                PSO_FIXED_X_FOR_F1F2, PSO_FIXED_N_FOR_F1F2
            )
        )
        print("=" * 60)
        for name in results:
            avg = results[name]["mean"]
            std = results[name]["std"]
            print("  {} -> среднее {} итер. (разброс {})".format(name, int(avg), int(std)))

    return results


def run_experiment_2_ga(verbose=True):
    results = {"BLX-alpha": {}, "Discrete": {}}
    for crossover in ["BLX-alpha", "Discrete"]:
        for pop_size in POP_SIZES:
            for mut_rate in MUTATION_RATES:
                config_name = "pop={} mut={}".format(pop_size, mut_rate)
                errors = []
                for _ in range(NUM_RUNS):
                    _, best_val, _, _ = run_ga(crossover, mut_rate, pop_size, max_iter=FIXED_ITERATIONS)
                    errors.append(abs(best_val - OPTIMAL_VALUE))
                avg, std = mean_std(errors)
                results[crossover][config_name] = {"mean": avg, "std": std}

    if verbose:
        print("")
        print("=" * 60)
        print("ЭКСПЕРИМЕНТ 2 (GA): какая точность за {} итераций".format(FIXED_ITERATIONS))
        print("=" * 60)
        print("")
        for crossover in ["BLX-alpha", "Discrete"]:
            print("GA {}:".format(crossover))
            for name in results[crossover]:
                avg = results[crossover][name]["mean"]
                std = results[crossover][name]["std"]
                print("  {} -> отклонение: {:.4f} (разброс {:.4f})".format(name, avg, std))
            print("")

    return results


def run_experiment_2_pso_x(verbose=True):
    results_by_x = {}
    for X_val in PSO_X_VALUES:
        results_by_x[X_val] = {}
        for n in POP_SIZES:
            config_name = "n={}".format(n)
            errors = []
            for _ in range(NUM_RUNS):
                _, best_val, _, _ = run_pso(PSO_BEST_F1, PSO_BEST_F2, X_val, n, max_iter=FIXED_ITERATIONS)
                errors.append(abs(best_val - OPTIMAL_VALUE))
            avg, std = mean_std(errors)
            results_by_x[X_val][config_name] = {"mean": avg, "std": std}

    if verbose:
        print("")
        print("=" * 60)
        print(
            "ЭКСПЕРИМЕНТ 2 (PSO-X,n): фиксируем f1={} f2={}, точность за {} итераций, меняем X и n".format(
                PSO_BEST_F1, PSO_BEST_F2, FIXED_ITERATIONS
            )
        )
        print("=" * 60)
        for X_val in PSO_X_VALUES:
            print("")
            print("PSO X={}:".format(X_val))
            for name in results_by_x[X_val]:
                avg = results_by_x[X_val][name]["mean"]
                std = results_by_x[X_val][name]["std"]
                print("  {} -> отклонение: {:.4f} (разброс {:.4f})".format(name, avg, std))

    return results_by_x


def run_experiment_2_pso_f1f2(verbose=True):
    results = {}
    for f1, f2 in PSO_F1F2_VALUES:
        config_name = "f1={} f2={}".format(f1, f2)
        errors = []
        for _ in range(NUM_RUNS):
            _, best_val, _, _ = run_pso(
                f1,
                f2,
                PSO_FIXED_X_FOR_F1F2,
                PSO_FIXED_N_FOR_F1F2,
                max_iter=FIXED_ITERATIONS,
            )
            errors.append(abs(best_val - OPTIMAL_VALUE))
        avg, std = mean_std(errors)
        results[config_name] = {"mean": avg, "std": std}

    if verbose:
        print("")
        print("=" * 60)
        print(
            "ЭКСПЕРИМЕНТ 2 (PSO-f1f2): фиксируем X={} и n={}, точность за {} итераций, меняем f1,f2".format(
                PSO_FIXED_X_FOR_F1F2, PSO_FIXED_N_FOR_F1F2, FIXED_ITERATIONS
            )
        )
        print("=" * 60)
        for name in results:
            avg = results[name]["mean"]
            std = results[name]["std"]
            print("  {} -> отклонение: {:.4f} (разброс {:.4f})".format(name, avg, std))

    return results


def run_experiment_3(verbose=True):
    results_ga = {}
    results_pso_x = {}
    results_pso_f1f2 = {}

    for cross_type in ["BLX-alpha", "Discrete"]:
        histories = []
        for _ in range(NUM_RUNS):
            _, _, _, history = run_ga(cross_type, 0.2, 100, max_iter=CONVERGENCE_ITERS)
            histories.append(history)
        results_ga[cross_type] = avg_history(histories)

    for X_val in PSO_X_VALUES:
        histories = []
        for _ in range(NUM_RUNS):
            _, _, _, history = run_pso(PSO_BEST_F1, PSO_BEST_F2, X_val, 100, max_iter=CONVERGENCE_ITERS)
            histories.append(history)
        results_pso_x[X_val] = avg_history(histories)

    for f1, f2 in PSO_F1F2_VALUES:
        key = "f1={} f2={}".format(f1, f2)
        histories = []
        for _ in range(NUM_RUNS):
            _, _, _, history = run_pso(f1, f2, PSO_FIXED_X_FOR_F1F2, 100, max_iter=CONVERGENCE_ITERS)
            histories.append(history)
        results_pso_f1f2[key] = avg_history(histories)

    if verbose:
        print("")
        print("=" * 60)
        print(
            "ЭКСПЕРИМЕНТ 3: кривые сходимости (n/pop=100, {} итераций, среднее по {} запускам)".format(
                CONVERGENCE_ITERS, NUM_RUNS
            )
        )
        print("=" * 60)
        print("")
        print("GA:")
        for name in results_ga:
            if results_ga[name]:
                print("  {} -> финал: {:.4f}".format(name, results_ga[name][-1]))
        print("")
        print("PSO (фиксируем f1={} f2={}, меняем X):".format(PSO_BEST_F1, PSO_BEST_F2))
        for X_val in PSO_X_VALUES:
            if results_pso_x[X_val]:
                print("  X={} -> финал: {:.4f}".format(X_val, results_pso_x[X_val][-1]))
        print("")
        print("PSO (фиксируем X={} и n=100, меняем f1,f2):".format(PSO_FIXED_X_FOR_F1F2))
        for key in results_pso_f1f2:
            if results_pso_f1f2[key]:
                print("  {} -> финал: {:.4f}".format(key, results_pso_f1f2[key][-1]))

    return results_ga, results_pso_x, results_pso_f1f2


def plot_ga_experiment_1(results_ga):
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    for ax, crossover, color in [(axes[0], "BLX-alpha", "#3498db"), (axes[1], "Discrete", "#2ecc71")]:
        table = results_ga[crossover]
        names = list(table.keys())
        means = [table[n]["mean"] for n in names]
        stds = [table[n]["std"] for n in names]
        ax.bar(range(len(names)), means, yerr=stds, capsize=3, color=color, alpha=0.85)
        ax.set_xticks(range(len(names)))
        ax.set_xticklabels(names, rotation=45, ha="right")
        ax.set_ylabel("Среднее число итераций")
        ax.set_title("GA {}: итерации до ε={}".format(crossover, EPSILON))
        ax.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    plt.savefig("ga_experiment_1_iterations.png", dpi=150)
    plt.close()


def plot_ga_experiment_2(results_ga):
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    for ax, crossover, color in [(axes[0], "BLX-alpha", "#3498db"), (axes[1], "Discrete", "#2ecc71")]:
        table = results_ga[crossover]
        names = list(table.keys())
        means = [table[n]["mean"] for n in names]
        stds = [table[n]["std"] for n in names]
        ax.bar(range(len(names)), means, yerr=stds, capsize=3, color=color, alpha=0.85)
        ax.set_xticks(range(len(names)))
        ax.set_xticklabels(names, rotation=45, ha="right")
        ax.set_ylabel("Отклонение от оптимума")
        ax.set_title("GA {}: точность за {} итераций".format(crossover, FIXED_ITERATIONS))
        ax.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    plt.savefig("ga_experiment_2_accuracy.png", dpi=150)
    plt.close()


def plot_ga_convergence(results_ga):
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(10, 6))
    for name, color in [("BLX-alpha", "#3498db"), ("Discrete", "#2ecc71")]:
        ax.plot(results_ga[name], label=name, color=color, linewidth=1.5)
    ax.axhline(y=OPTIMAL_VALUE, color="black", linestyle="--", label="Оптимум")
    ax.set_xlabel("Итерация")
    ax.set_ylabel("Лучшее значение")
    ax.set_title("GA: кривые сходимости")
    ax.grid(alpha=0.3)
    ax.legend()
    plt.tight_layout()
    plt.savefig("ga_convergence.png", dpi=150)
    plt.close()


def plot_pso_experiment_1(results_by_x):
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, len(PSO_X_VALUES), figsize=(5 * len(PSO_X_VALUES), 5))
    if len(PSO_X_VALUES) == 1:
        axes = [axes]
    colors = ["#e74c3c", "#f39c12", "#9b59b6"]
    for i, X_val in enumerate(PSO_X_VALUES):
        table = results_by_x[X_val]
        names = list(table.keys())
        means = [table[n]["mean"] for n in names]
        stds = [table[n]["std"] for n in names]
        axes[i].bar(range(len(names)), means, yerr=stds, capsize=3, color=colors[i % len(colors)], alpha=0.85)
        axes[i].set_xticks(range(len(names)))
        axes[i].set_xticklabels(names)
        axes[i].set_ylabel("Среднее число итераций")
        axes[i].set_title(
            "PSO X={}: итерации до ε={} (f1={} f2={})".format(X_val, EPSILON, PSO_BEST_F1, PSO_BEST_F2)
        )
        axes[i].grid(axis="y", alpha=0.3)
    plt.tight_layout()
    plt.savefig("pso_experiment_1_iterations.png", dpi=150)
    plt.close()


def plot_pso_experiment_2(results_by_x):
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, len(PSO_X_VALUES), figsize=(5 * len(PSO_X_VALUES), 5))
    if len(PSO_X_VALUES) == 1:
        axes = [axes]
    colors = ["#e74c3c", "#f39c12", "#9b59b6"]
    for i, X_val in enumerate(PSO_X_VALUES):
        table = results_by_x[X_val]
        names = list(table.keys())
        means = [table[n]["mean"] for n in names]
        stds = [table[n]["std"] for n in names]
        axes[i].bar(range(len(names)), means, yerr=stds, capsize=3, color=colors[i % len(colors)], alpha=0.85)
        axes[i].set_xticks(range(len(names)))
        axes[i].set_xticklabels(names)
        axes[i].set_ylabel("Отклонение от оптимума")
        axes[i].set_title(
            "PSO X={}: точность за {} (f1={} f2={})".format(X_val, FIXED_ITERATIONS, PSO_BEST_F1, PSO_BEST_F2)
        )
        axes[i].grid(axis="y", alpha=0.3)
    plt.tight_layout()
    plt.savefig("pso_experiment_2_accuracy.png", dpi=150)
    plt.close()


def plot_pso_convergence(results_pso_x):
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(10, 6))
    colors = ["#e74c3c", "#f39c12", "#9b59b6"]
    for i, X_val in enumerate(PSO_X_VALUES):
        ax.plot(results_pso_x[X_val], label="X={}".format(X_val), color=colors[i % len(colors)], linewidth=1.5)
    ax.axhline(y=OPTIMAL_VALUE, color="black", linestyle="--", label="Оптимум")
    ax.set_xlabel("Итерация")
    ax.set_ylabel("Лучшее значение")
    ax.set_title("PSO: кривые сходимости (f1={} f2={})".format(PSO_BEST_F1, PSO_BEST_F2))
    ax.grid(alpha=0.3)
    ax.legend()
    plt.tight_layout()
    plt.savefig("pso_convergence.png", dpi=150)
    plt.close()


def plot_pso_params_experiment_1(results):
    import matplotlib.pyplot as plt

    names = list(results.keys())
    means = [results[n]["mean"] for n in names]
    stds = [results[n]["std"] for n in names]
    fig, ax = plt.subplots(figsize=(14, 6))
    ax.bar(range(len(names)), means, yerr=stds, capsize=3, color="#16a085", alpha=0.85)
    ax.set_xticks(range(len(names)))
    ax.set_xticklabels(names, rotation=45, ha="right")
    ax.set_ylabel("Среднее число итераций")
    ax.set_title(
        "PSO (X={}, n={}): влияние f1,f2 на итерации до ε={}".format(
            PSO_FIXED_X_FOR_F1F2, PSO_FIXED_N_FOR_F1F2, EPSILON
        )
    )
    ax.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    plt.savefig("pso_params_experiment_1_iterations.png", dpi=150)
    plt.close()


def plot_pso_params_experiment_2(results):
    import matplotlib.pyplot as plt

    names = list(results.keys())
    means = [results[n]["mean"] for n in names]
    stds = [results[n]["std"] for n in names]
    fig, ax = plt.subplots(figsize=(14, 6))
    ax.bar(range(len(names)), means, yerr=stds, capsize=3, color="#16a085", alpha=0.85)
    ax.set_xticks(range(len(names)))
    ax.set_xticklabels(names, rotation=45, ha="right")
    ax.set_ylabel("Отклонение от оптимума")
    ax.set_title(
        "PSO (X={}, n={}): влияние f1,f2 на точность за {}".format(
            PSO_FIXED_X_FOR_F1F2, PSO_FIXED_N_FOR_F1F2, FIXED_ITERATIONS
        )
    )
    ax.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    plt.savefig("pso_params_experiment_2_accuracy.png", dpi=150)
    plt.close()


def plot_pso_params_convergence(results_pso_f1f2):
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(10, 6))
    for name in results_pso_f1f2:
        ax.plot(results_pso_f1f2[name], label=name, linewidth=1.5)
    ax.axhline(y=OPTIMAL_VALUE, color="black", linestyle="--", label="Оптимум")
    ax.set_xlabel("Итерация")
    ax.set_ylabel("Лучшее значение")
    ax.set_title("PSO: кривые сходимости (X={}, разные f1,f2)".format(PSO_FIXED_X_FOR_F1F2))
    ax.grid(alpha=0.3)
    ax.legend()
    plt.tight_layout()
    plt.savefig("pso_params_convergence.png", dpi=150)
    plt.close()


def plot_ga_vs_pso(ga_conv, pso_conv):
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(10, 6))
    ga_avg = ga_conv["BLX-alpha"]
    pso_avg = pso_conv[BEST_PSO_X_FOR_COMPARISON]
    ax.plot(ga_avg, label="GA (BLX-alpha)", color="#3498db", linewidth=1.5)
    ax.plot(
        pso_avg,
        label="PSO (X={}, f1={} f2={})".format(BEST_PSO_X_FOR_COMPARISON, PSO_BEST_F1, PSO_BEST_F2),
        color="#e74c3c",
        linewidth=1.5,
    )
    ax.axhline(y=OPTIMAL_VALUE, color="black", linestyle="--", label="Оптимум")
    ax.set_xlabel("Итерация")
    ax.set_ylabel("Лучшее значение")
    ax.set_title("Сравнение лучших GA и PSO (PSO выбран X={})".format(BEST_PSO_X_FOR_COMPARISON))
    ax.grid(alpha=0.3)
    ax.legend()
    plt.tight_layout()
    plt.savefig("ga_vs_pso.png", dpi=150)
    plt.close()


def main():
    mode = "all"
    if len(sys.argv) >= 2:
        mode = sys.argv[1].strip().lower()
    if mode not in ["all", "stats", "plots"]:
        print("Usage: python research.py [all|stats|plots]")
        return

    verbose = mode in ["all", "stats"]
    do_plots = mode in ["all", "plots"]

    if verbose:
        print("")
        print("=" * 60)
        print("ИССЛЕДОВАНИЕ GA И PSO")
        print("=" * 60)
        print("Функция Eggholder, оптимум = {}".format(OPTIMAL_VALUE))
        print("Каждая конфигурация тестируется {} раз".format(NUM_RUNS))

    ga_1 = run_experiment_1_ga(verbose=verbose)
    pso_x_1 = run_experiment_1_pso_x(verbose=verbose)
    pso_f_1 = run_experiment_1_pso_f1f2(verbose=verbose)

    ga_2 = run_experiment_2_ga(verbose=verbose)
    pso_x_2 = run_experiment_2_pso_x(verbose=verbose)
    pso_f_2 = run_experiment_2_pso_f1f2(verbose=verbose)

    ga_conv, pso_x_conv, pso_f_conv = run_experiment_3(verbose=verbose)

    if do_plots:
        plot_ga_experiment_1(ga_1)
        plot_ga_experiment_2(ga_2)
        plot_ga_convergence(ga_conv)
        plot_pso_experiment_1(pso_x_1)
        plot_pso_experiment_2(pso_x_2)
        plot_pso_convergence(pso_x_conv)
        plot_pso_params_experiment_1(pso_f_1)
        plot_pso_params_experiment_2(pso_f_2)
        plot_pso_params_convergence(pso_f_conv)
        plot_ga_vs_pso(ga_conv, pso_x_conv)

    if verbose:
        print("")
        print("=" * 60)
        print("ЛУЧШИЕ РЕЗУЛЬТАТЫ (по средним значениям)")
        print("=" * 60)

        best_ga_blx = sorted(ga_1["BLX-alpha"].items(), key=lambda x: x[1]["mean"])[0]
        best_ga_dis = sorted(ga_1["Discrete"].items(), key=lambda x: x[1]["mean"])[0]
        print("")
        print("Эксперимент 1 (меньше итераций = лучше):")
        print("  GA BLX-alpha лучший: {} = {}".format(best_ga_blx[0], int(best_ga_blx[1]["mean"])))
        print("  GA Discrete лучший: {} = {}".format(best_ga_dis[0], int(best_ga_dis[1]["mean"])))

        best_pso_x = None
        for X_val in PSO_X_VALUES:
            for name in pso_x_1[X_val]:
                cand = (pso_x_1[X_val][name]["mean"], X_val, name)
                if best_pso_x is None or cand < best_pso_x:
                    best_pso_x = cand
        print(
            "  PSO (фиксируем f1={} f2={}, меняем X,n) лучший по среднему: X={} {} = {}".format(
                PSO_BEST_F1, PSO_BEST_F2, best_pso_x[1], best_pso_x[2], int(best_pso_x[0])
            )
        )
        best_pso_f = sorted(pso_f_1.items(), key=lambda x: x[1]["mean"])[0]
        print(
            "  PSO (фиксируем X={} n={}, меняем f1,f2) лучший: {} = {}".format(
                PSO_FIXED_X_FOR_F1F2,
                PSO_FIXED_N_FOR_F1F2,
                best_pso_f[0],
                int(best_pso_f[1]["mean"]),
            )
        )

        best_ga_blx_2 = sorted(ga_2["BLX-alpha"].items(), key=lambda x: x[1]["mean"])[0]
        best_ga_dis_2 = sorted(ga_2["Discrete"].items(), key=lambda x: x[1]["mean"])[0]
        print("")
        print("Эксперимент 2 (меньше отклонение = лучше):")
        print("  GA BLX-alpha лучший: {} = {:.4f}".format(best_ga_blx_2[0], best_ga_blx_2[1]["mean"]))
        print("  GA Discrete лучший: {} = {:.4f}".format(best_ga_dis_2[0], best_ga_dis_2[1]["mean"]))

        best_pso_x_2 = None
        for X_val in PSO_X_VALUES:
            for name in pso_x_2[X_val]:
                cand = (pso_x_2[X_val][name]["mean"], X_val, name)
                if best_pso_x_2 is None or cand < best_pso_x_2:
                    best_pso_x_2 = cand
        print(
            "  PSO (фиксируем f1={} f2={}, меняем X,n) лучший по среднему: X={} {} = {:.4f}".format(
                PSO_BEST_F1, PSO_BEST_F2, best_pso_x_2[1], best_pso_x_2[2], best_pso_x_2[0]
            )
        )
        best_pso_f_2 = sorted(pso_f_2.items(), key=lambda x: x[1]["mean"])[0]
        print(
            "  PSO (фиксируем X={} n={}, меняем f1,f2) лучший: {} = {:.4f}".format(
                PSO_FIXED_X_FOR_F1F2,
                PSO_FIXED_N_FOR_F1F2,
                best_pso_f_2[0],
                best_pso_f_2[1]["mean"],
            )
        )

        print("")
        print("Для графика сравнения выбрано PSO с X={} (как требуется).".format(BEST_PSO_X_FOR_COMPARISON))


if __name__ == "__main__":
    main()
