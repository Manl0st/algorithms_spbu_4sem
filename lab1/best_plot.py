import math
import os
import random

import matplotlib.pyplot as plt

OPTIMAL_VALUE = -959.6407
SCOPE = (-512.0, 512.0)


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


def run_ga(crossover_type, mut_rate, pop_size, max_iter):
    best_chromosome = None
    best_fitness_value = None
    pop = [[random.uniform(*SCOPE) for _ in range(2)] for _ in range(pop_size)]
    best_fitness_history = []

    for _ in range(max_iter):
        new_pop = pop[:]
        while len(new_pop) < pop_size * 2:
            chrom1 = selection(pop)
            chrom2 = selection(pop)
            chrom3, chrom4 = blx_alpha(chrom1, chrom2)
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

    return best_fitness_history


def run_pso(f1, f2, particle_size, max_iter):
    X = 1.0
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
    for _ in range(max_iter):
        for p in particle_list:
            move(p)
        best_history.append(fitness_gbest)

    return best_history


def main():
    iterations = int(os.getenv("CONVERGENCE_ITERS", "500"))
    runs = int(os.getenv("NUM_RUNS", "25"))

    ga_histories = []
    for _ in range(runs):
        ga_histories.append(run_ga("BLX-alpha", 0.2, 100, iterations))

    pso_histories = []
    for _ in range(runs):
        pso_histories.append(run_pso(2.05, 2.05, 100, iterations))

    ga_history = []
    for i in range(iterations):
        values = []
        for h in ga_histories:
            values.append(h[i])
        ga_history.append(sum(values) / len(values))

    pso_history = []
    for i in range(iterations):
        values = []
        for h in pso_histories:
            values.append(h[i])
        pso_history.append(sum(values) / len(values))

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(ga_history, label="GA BLX-alpha (pop=100, mut=0.2)", linewidth=1.5)
    ax.plot(pso_history, label="PSO X=1.0 (n=100)", linewidth=1.5)
    ax.axhline(y=OPTIMAL_VALUE, color="black", linestyle="--", label="минимум")
    ax.set_xlabel("Итерация")
    ax.set_ylabel("Лучшее значение")
    ax.set_title("Сходимость лучших GA и PSO ")
    ax.grid(alpha=0.3)
    ax.legend()
    plt.tight_layout()
    plt.savefig("best_ga_vs_pso.png", dpi=150)
    plt.close()


if __name__ == "__main__":
    main()
