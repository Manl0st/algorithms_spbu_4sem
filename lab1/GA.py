import math
import random
import tkinter as tk
from tkinter import ttk
from matplotlib.figure import Figure
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

alpha = 0.5
scope = [-512, 512]
selection_size = 4
round_value = 10

def fitness_function(x, y):
    return -(y + 47) * math.sin(math.sqrt(math.fabs(x / 2 + (y + 47)))) - x * math.sin(math.sqrt(math.fabs(x - (y + 47))))

def fitness_function_value(chromosome):
    return round(fitness_function(chromosome[0], chromosome[1]), round_value)

def mutate(chromosome, rate, scope):
    if random.random() < rate:
        for i in range(len(chromosome)):
            chromosome[i] = random.uniform(*scope)
    return chromosome

def selection(population, size):
    competition_list = []
    while len(competition_list) < size:
        competition_list.append(random.choice(population))
    best_chromosome = None
    best_function_value = None
    for selected in competition_list:
        selected_value = fitness_function_value(selected)
        if best_chromosome is None or selected_value < best_function_value:
           best_function_value = selected_value
           best_chromosome = selected
    return best_chromosome

def discrete_recombination(chrom1, chrom2):
    child1 = [random.choice([chrom1[0], chrom2[0]]), random.choice([chrom1[1], chrom2[1]])]
    child2 = [random.choice([chrom1[0], chrom2[0]]), random.choice([chrom1[1], chrom2[1]])]
    return [child1, child2]
    
def BLX_alpha(chrom1, chrom2):
    new_chromosomes = [[], []]
    for i in range(2):
        for j in range(2):
            d = abs(chrom1[j]-chrom2[j])
            new_val = random.uniform(min(chrom1[j], chrom2[j]) - alpha * d, max(chrom1[j], chrom2[j]) + alpha * d)
            new_val = max(scope[0], min(scope[1], new_val))
            new_chromosomes[i].append(new_val)
    return new_chromosomes

def run_genetic_algorithm(crossover_type, mut_rate, pop_size, max_iter):
    best_chromosome = None
    best_fitness_value = None
    iter_count = 0
    pop = [[random.uniform(*scope) for i in range(2)] for j in range(pop_size)]
    best_fitness_history = []
    avg_fitness_history = []

    while iter_count < max_iter:
        new_pop = pop[:]
        while len(new_pop) < pop_size * 2:
            chrom1 = selection(pop, selection_size)
            chrom2 = selection(pop, selection_size)
            if crossover_type == "Промежуточная комбинация":
                chrom3, chrom4 = BLX_alpha(chrom1, chrom2)
            else:
                chrom3, chrom4 = discrete_recombination(chrom1, chrom2) 
            mutate(chrom3, mut_rate, scope)
            mutate(chrom4, mut_rate, scope)
            new_pop.append(chrom3)
            new_pop.append(chrom4)
            
        pop = sorted(new_pop, key=lambda chrom : fitness_function_value(chrom))[:pop_size]
        fit_val = fitness_function_value(pop[0])
        if best_chromosome is None or best_fitness_value > fit_val:
            best_fitness_value = fit_val
            best_chromosome = pop[0]
        generation_average_fitness = round(sum(map(fitness_function_value, pop)) / pop_size, round_value)
        best_fitness_history.append(best_fitness_value)
        avg_fitness_history.append(generation_average_fitness)
        iter_count += 1

    return best_chromosome, best_fitness_value, best_fitness_history, avg_fitness_history


class GeneticAlgorithmApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Генетический алгоритм")
        self.root.geometry("1000x600")
        self.left_frame = tk.Frame(root, width=300, padx=10, pady=10)
        self.left_frame.pack(side=tk.LEFT, fill=tk.Y)
        self.right_frame = tk.Frame(root, padx=10, pady=10)
        self.right_frame.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True)
        self.setup_ui()
        self.setup_plots()

    def setup_ui(self):
        tk.Label(self.left_frame, text="Тип кроссинговера:").pack(anchor="w", pady=(0, 5))
        self.crossover_var = tk.StringVar(value="Промежуточная комбинация")
        crossover_cb = ttk.Combobox(self.left_frame, textvariable=self.crossover_var, state="readonly")
        crossover_cb['values'] = ("Промежуточная комбинация", "Дискретная рекомбинация")
        crossover_cb.pack(fill=tk.X, pady=(0, 15))
        tk.Label(self.left_frame, text="Вероятность мутации:").pack(anchor="w")
        self.mut_rate_var = tk.DoubleVar(value=0.2)
        tk.Entry(self.left_frame, textvariable=self.mut_rate_var).pack(fill=tk.X, pady=(0, 15))
        tk.Label(self.left_frame, text="Количество хромосом:").pack(anchor="w")
        self.pop_size_var = tk.IntVar(value=100)
        tk.Entry(self.left_frame, textvariable=self.pop_size_var).pack(fill=tk.X, pady=(0, 15))
        tk.Label(self.left_frame, text="Количество поколений:").pack(anchor="w")
        self.max_iter_var = tk.IntVar(value=500)
        tk.Entry(self.left_frame, textvariable=self.max_iter_var).pack(fill=tk.X, pady=(0, 20))
        tk.Button(self.left_frame, text="Запустить алгоритм", command=self.run_algo, bg="lightblue").pack(fill=tk.X, pady=(0, 30))
        tk.Label(self.left_frame, text="Результаты:", font=("Helvetica", 12, "bold")).pack(anchor="w")
        self.res_coords_label = tk.Label(self.left_frame, text="Координаты лучшего решения:\n-", justify="left")
        self.res_coords_label.pack(anchor="w", pady=5)
        self.res_fitness_label = tk.Label(self.left_frame, text="Значение функции:\n-", justify="left")
        self.res_fitness_label.pack(anchor="w", pady=5)

    def setup_plots(self):
        self.figure = Figure(figsize=(6, 5), dpi=100)
        self.ax1 = self.figure.add_subplot(211)
        self.ax2 = self.figure.add_subplot(212)
        self.figure.tight_layout(pad=3.0)
        self.canvas = FigureCanvasTkAgg(self.figure, master=self.right_frame)
        self.canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)

    def run_algo(self):
        c_type = self.crossover_var.get()
        m_rate = self.mut_rate_var.get()
        p_size = self.pop_size_var.get()
        m_iter = self.max_iter_var.get()
        best_chrom, best_fit, history_best, history_avg = run_genetic_algorithm(c_type, m_rate, p_size, m_iter)
        coords_str = f"[{round(best_chrom[0], 2)}, {round(best_chrom[1], 2)}]"
        self.res_coords_label.config(text=f"Координаты лучшего решения:\n{coords_str}")
        self.res_fitness_label.config(text=f"Значение функции:\n{best_fit}")
        self.ax1.clear()
        self.ax1.plot(range(m_iter), history_best, color='blue')
        self.ax1.set_xlabel("Номер поколения")
        self.ax1.set_ylabel("Лучшее значение функции")
        self.ax1.grid(True)
        self.ax2.clear()
        self.ax2.plot(range(m_iter), history_avg, color='green')
        self.ax2.set_xlabel("Номер поколения")
        self.ax2.set_ylabel("Среднее значение функции")
        self.ax2.grid(True)
        self.figure.tight_layout(pad=3.0)
        self.canvas.draw()

if __name__ == "__main__":
    root = tk.Tk()
    app = GeneticAlgorithmApp(root)
    root.mainloop()