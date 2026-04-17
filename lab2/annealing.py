import math
import random
import tkinter as tk
from tkinter import ttk
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure
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

def route_cost(matrix, path):
    total = 0
    for i in range(len(path)):
        weight = matrix[path[i]][path[(i + 1) % len(path)]]
        if weight == INF:
            return INF
        total += weight
    return total

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
    initial_cost = current_cost
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
        "best_path": best_path,
        "best_cost": best_cost,
        "initial_cost": initial_cost,
        "steps": step,
        "history_cost": history_cost,
        "history_temp": history_temp,
    }

class TSPGui:
    def __init__(self, root):
        self.root = root
        self.root.title("TSP Solver - Simulated Annealing")
        self.root.geometry("900x600")
        
        self.graphs = {
            0: ("Built-in (6)", MATRIX_1GRAPH),
            1: ("BERLIN52 (52)", BERLIN52),
            2: ("WORLD666 (666)", WORLD666),
        }
        
        self.setup_ui()
    
    def setup_ui(self):
        # Левая панель - параметры
        left = ttk.Frame(self.root)
        left.pack(side=tk.LEFT, fill=tk.BOTH, padx=10, pady=10)
        
        # График
        ttk.Label(left, text="График:", font=("Arial", 10, "bold")).pack(anchor=tk.W)
        self.graph_var = tk.IntVar(value=1)
        for key, (name, _) in self.graphs.items():
            ttk.Radiobutton(left, text=name, variable=self.graph_var, value=key).pack(anchor=tk.W)
        
        ttk.Separator(left, orient=tk.HORIZONTAL).pack(fill=tk.X, pady=10)
        
        # Режим
        ttk.Label(left, text="Режим охлаждения:", font=("Arial", 10, "bold")).pack(anchor=tk.W)
        self.mode_var = tk.StringVar(value="classic")
        ttk.Radiobutton(left, text="Classic (T=T×α)", variable=self.mode_var, value="classic").pack(anchor=tk.W)
        ttk.Radiobutton(left, text="Cauchy (T=T₀/(1+n))", variable=self.mode_var, value="cauchy").pack(anchor=tk.W)
        
        ttk.Separator(left, orient=tk.HORIZONTAL).pack(fill=tk.X, pady=10)
        
        # Параметры
        ttk.Label(left, text="Параметры:", font=("Arial", 10, "bold")).pack(anchor=tk.W)
        
        ttk.Label(left, text="T_START:").pack(anchor=tk.W)
        self.t_start = ttk.Entry(left, width=15)
        self.t_start.insert(0, "100.0")
        self.t_start.pack(fill=tk.X, pady=2)
        
        ttk.Label(left, text="T_MIN:").pack(anchor=tk.W)
        self.t_min = ttk.Entry(left, width=15)
        self.t_min.insert(0, "0.089")
        self.t_min.pack(fill=tk.X, pady=2)
        
        ttk.Label(left, text="ALPHA (Classic):").pack(anchor=tk.W)
        self.alpha = ttk.Entry(left, width=15)
        self.alpha.insert(0, "0.95")
        self.alpha.pack(fill=tk.X, pady=2)
        
        ttk.Label(left, text="ITERS_PER_T:").pack(anchor=tk.W)
        self.iters = ttk.Entry(left, width=15)
        self.iters.insert(0, "30")
        self.iters.pack(fill=tk.X, pady=2)
        
        ttk.Separator(left, orient=tk.HORIZONTAL).pack(fill=tk.X, pady=10)
        
        # Кнопка запуска
        self.btn = ttk.Button(left, text="▶ Запустить", command=self.run)
        self.btn.pack(fill=tk.X, pady=10)
        
        # Результаты
        self.result_text = tk.Text(left, height=15, width=35, font=("Courier", 8))
        self.result_text.pack(fill=tk.BOTH, expand=True)
        
        # Правая панель - графики
        self.right = ttk.Frame(self.root)
        self.right.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True, padx=10, pady=10)
    
    def run(self):
        try:
            graph_idx = self.graph_var.get()
            graph_name, matrix = self.graphs[graph_idx]
            mode = self.mode_var.get()
            t_start = float(self.t_start.get())
            t_min = float(self.t_min.get())
            alpha = float(self.alpha.get()) if mode == "classic" else 0.95
            iters = int(self.iters.get())
            
            # Запустить алгоритм
            self.btn.config(state=tk.DISABLED)
            self.result_text.delete(1.0, tk.END)
            self.result_text.insert(tk.END, "⏳ Вычисляю...\n")
            self.root.update()
            
            result = simulated_annealing(matrix, mode, t_start, t_min, alpha, iters)
            
            # Вывести результаты
            text = f"График: {graph_name}\n"
            text += f"Режим: {mode.upper()}\n"
            text += f"T_START: {t_start}\n"
            text += f"T_MIN: {t_min}\n"
            text += f"ALPHA: {alpha}\n"
            text += f"ITERS: {iters}\n\n"
            text += f"Начально: {result['initial_cost']:.2f}\n"
            text += f"Лучше: {result['best_cost']:.2f}\n"
            imp = (result['initial_cost'] - result['best_cost']) / result['initial_cost'] * 100
            text += f"Улучшение: {imp:.1f}%\n"
            text += f"Шагов: {result['steps']}\n"
            
            self.result_text.delete(1.0, tk.END)
            self.result_text.insert(tk.END, text)
            
            # Построить графики
            self.plot_results(result)
            
            self.btn.config(state=tk.NORMAL)
        except Exception as e:
            self.result_text.delete(1.0, tk.END)
            self.result_text.insert(tk.END, f"Ошибка: {e}")
            self.btn.config(state=tk.NORMAL)
    
    def plot_results(self, result):
        # Очистить старые графики
        for widget in self.right.winfo_children():
            widget.destroy()
        
        # Создать фигуру
        fig = Figure(figsize=(5, 5), dpi=80)
        ax1 = fig.add_subplot(121)
        ax2 = fig.add_subplot(122)
        
        steps = list(range(len(result['history_cost'])))
        
        # График 1: Сходимость
        ax1.plot(steps, result['history_cost'], 'b-', linewidth=1)
        ax1.set_xlabel('Шаг')
        ax1.set_ylabel('Стоимость')
        ax1.set_title('Сходимость')
        ax1.grid(True, alpha=0.3)
        
        # График 2: Температура
        ax2.plot(steps, result['history_temp'], 'r-', linewidth=1)
        ax2.set_xlabel('Шаг')
        ax2.set_ylabel('T')
        ax2.set_title('Температура')
        ax2.grid(True, alpha=0.3)
        
        fig.tight_layout()
        
        # Встроить в GUI
        canvas = FigureCanvasTkAgg(fig, master=self.right)
        canvas.draw()
        canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)

if __name__ == "__main__":
    root = tk.Tk()
    app = TSPGui(root)
    root.mainloop()
