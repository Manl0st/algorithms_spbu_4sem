import math
import random
import tkinter as tk
from tkinter import ttk
from matplotlib.figure import Figure
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

def fitness_function(x, y):
    return -(y + 47) * math.sin(math.sqrt(math.fabs(x / 2 + (y + 47)))) - x * math.sin(math.sqrt(math.fabs(x - (y + 47))))

def run_pso(use_constriction, f1, f2, particle_size, max_iter):
    f = f1 + f2
    k = 1.0
    if use_constriction and f > 4:
        X = (2 * k) / abs(2 - f - math.sqrt(math.pow(f, 2) - 4 * f))
    else:
        X = 1.0
    gbest = [random.uniform(-512, 512), random.uniform(-512, 512)]
    fitness_gbest = fitness_function(gbest[0], gbest[1])
    particle_list = []

    class patricle:
        def __init__(self, x, y, v0):
            self.fitness = fitness_function(x, y)
            self.position = [x, y]
            self.pbest = [x, y]
            self.fitness_pbest = self.fitness
            self.v = v0

    def move(p, X):
        nonlocal gbest, fitness_gbest
        r1 = random.uniform(0, 1)
        r2 = random.uniform(0, 1)
        v_x = X * (p.v[0] + f1 * r1 * (p.pbest[0] - p.position[0]) + f2 * r2 * (gbest[0] - p.position[0]))
        v_y = X * (p.v[1] + f1 * r1 * (p.pbest[1] - p.position[1]) + f2 * r2 * (gbest[1] - p.position[1]))
        p.v = [v_x, v_y]
        p.position[0] += v_x
        p.position[1] += v_y
        p.position[0] = min(512, max(-512, p.position[0]))
        p.position[1] = min(512, max(-512, p.position[1]))  
        p.fitness = fitness_function(p.position[0], p.position[1])
        if p.fitness < p.fitness_pbest:
            p.fitness_pbest = p.fitness
            p.pbest = p.position[:] 
        if p.fitness < fitness_gbest:
            fitness_gbest = p.fitness
            gbest = p.position[:]
        return p

    for _ in range(particle_size):
        v_x = random.uniform(-512, 512)
        v_y = random.uniform(-512, 512)
        v = [v_x , v_y]
        p = patricle(random.uniform(-512, 512), random.uniform(-512, 512), v)
        particle_list.append(p)
    best_history = []
    avg_history = []
    for _ in range(max_iter):
        for p in particle_list:
            move(p, X)
        best_history.append(fitness_gbest)
        avg_history.append(sum(p.fitness for p in particle_list) / particle_size)
    return gbest, fitness_gbest, best_history, avg_history

class PSOApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Роевой алгоритм")
        self.root.geometry("1000x600")
        self.left_frame = tk.Frame(root, width=300, padx=10, pady=10)
        self.left_frame.pack(side=tk.LEFT, fill=tk.Y)
        self.right_frame = tk.Frame(root, padx=10, pady=10)
        self.right_frame.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True)
        self.setup_ui()
        self.setup_plots()

    def setup_ui(self):
        tk.Label(self.left_frame, text="Модификация:").pack(anchor="w")
        self.mode_var = tk.StringVar(value="С коэффициентом сжатия")
        mode_cb = ttk.Combobox(self.left_frame, textvariable=self.mode_var, state="readonly")
        mode_cb['values'] = ("С коэффициентом сжатия", "Без коэффициента")
        mode_cb.pack(fill=tk.X, pady=(0, 15))
        tk.Label(self.left_frame, text="Коэфф. собств. лучшего значения:").pack(anchor="w")
        self.f1_var = tk.DoubleVar(value=2.05)
        tk.Entry(self.left_frame, textvariable=self.f1_var).pack(fill=tk.X, pady=(0, 15))
        tk.Label(self.left_frame, text="Коэфф. глоб. лучшего значения:").pack(anchor="w")
        self.f2_var = tk.DoubleVar(value=2.05)
        tk.Entry(self.left_frame, textvariable=self.f2_var).pack(fill=tk.X, pady=(0, 15))
        tk.Label(self.left_frame, text="Количество частиц:").pack(anchor="w")
        self.p_size_var = tk.IntVar(value=100)
        tk.Entry(self.left_frame, textvariable=self.p_size_var).pack(fill=tk.X, pady=(0, 15))
        tk.Label(self.left_frame, text="Количество итераций:").pack(anchor="w")
        self.max_iter_var = tk.IntVar(value=500)
        tk.Entry(self.left_frame, textvariable=self.max_iter_var).pack(fill=tk.X, pady=(0, 20))
        tk.Button(self.left_frame, text="Запустить алгоритм", command=self.run_algo, bg="lightblue").pack(fill=tk.X, pady=(0, 30))
        tk.Label(self.left_frame, text="Результаты:", font=("Helvetica", 12, "bold")).pack(anchor="w")
        self.res_coords_label = tk.Label(self.left_frame, text="Координаты лучшей частицы:\n-", justify="left")
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
        use_constr = self.mode_var.get() == "С коэффициентом сжатия"
        f1 = self.f1_var.get()
        f2 = self.f2_var.get()
        p_size = self.p_size_var.get()
        m_iter = self.max_iter_var.get()
        best_pos, best_fit, hist_best, hist_avg = run_pso(use_constr, f1, f2, p_size, m_iter)
        self.res_coords_label.config(text=f"Координаты лучшей частицы:\n[{round(best_pos[0], 5)}, {round(best_pos[1], 5)}]")
        self.res_fitness_label.config(text=f"Значение функции:\n{round(best_fit, 10)}")
        self.ax1.clear()
        self.ax1.plot(range(m_iter), hist_best, color='blue')
        self.ax1.set_xlabel("Номер итерации")
        self.ax1.set_ylabel("Лучшее значение функции")
        self.ax1.grid(True)
        self.ax2.clear()
        self.ax2.plot(range(m_iter), hist_avg, color='green')
        self.ax2.set_xlabel("Номер итерации")
        self.ax2.set_ylabel("Среднее значение функции")
        self.ax2.grid(True)
        self.figure.tight_layout(pad=3.0)
        self.canvas.draw()

if __name__ == "__main__":
    root = tk.Tk()
    app = PSOApp(root)
    root.mainloop()