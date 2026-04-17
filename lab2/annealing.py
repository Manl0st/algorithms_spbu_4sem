import math
import random
import time
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
        self.root.title("TSP Solver")
        self.root.geometry("1000x700")
        self.root.configure(bg="#f0f0f0")
        
        self.graphs = {
            0: ("Built-in (6)", MATRIX_1GRAPH),
            1: ("BERLIN52 (52)", BERLIN52),
            2: ("WORLD666 (666)", WORLD666),
        }
        
        self.setup_ui()
    
    def setup_ui(self):
        # Главный контейнер
        main = ttk.Frame(self.root)
        main.pack(fill=tk.BOTH, expand=True, padx=12, pady=12)
        
        # Левая панель - параметры
        left = ttk.LabelFrame(main, text="Параметры", padding=15)
        left.pack(side=tk.LEFT, fill=tk.BOTH, padx=(0, 10))
        
        # График
        ttk.Label(left, text="Граф:", font=("Arial", 10, "bold")).pack(anchor=tk.W, pady=(0, 5))
        self.graph_var = tk.IntVar(value=1)
        for key, (name, _) in self.graphs.items():
            ttk.Radiobutton(left, text=name, variable=self.graph_var, value=key).pack(anchor=tk.W)
        
        ttk.Separator(left, orient=tk.HORIZONTAL).pack(fill=tk.X, pady=12)
        
        # Режим охлаждения - комбобокс
        ttk.Label(left, text="Режим охлаждения:", font=("Arial", 10, "bold")).pack(anchor=tk.W, pady=(0, 5))
        self.mode_var = tk.StringVar(value="Classic")
        mode_combo = ttk.Combobox(left, textvariable=self.mode_var, 
                                  values=["Classic", "Cauchy"], state="readonly", width=20)
        mode_combo.pack(fill=tk.X)
        
        ttk.Separator(left, orient=tk.HORIZONTAL).pack(fill=tk.X, pady=12)
        
        # Параметры
        ttk.Label(left, text="Параметры:", font=("Arial", 10, "bold")).pack(anchor=tk.W, pady=(0, 5))
        
        ttk.Label(left, text="T_START:").pack(anchor=tk.W, pady=(3, 0))
        self.t_start = ttk.Entry(left, width=20)
        self.t_start.insert(0, "100.0")
        self.t_start.pack(fill=tk.X, pady=(0, 8))
        
        ttk.Label(left, text="T_MIN:").pack(anchor=tk.W, pady=(3, 0))
        self.t_min = ttk.Entry(left, width=20)
        self.t_min.insert(0, "0.089")
        self.t_min.pack(fill=tk.X, pady=(0, 8))
        
        ttk.Label(left, text="ALPHA:").pack(anchor=tk.W, pady=(3, 0))
        self.alpha = ttk.Entry(left, width=20)
        self.alpha.insert(0, "0.95")
        self.alpha.pack(fill=tk.X, pady=(0, 8))
        
        ttk.Label(left, text="ITERS_PER_T:").pack(anchor=tk.W, pady=(3, 0))
        self.iters = ttk.Entry(left, width=20)
        self.iters.insert(0, "30")
        self.iters.pack(fill=tk.X, pady=(0, 12))
        
        ttk.Separator(left, orient=tk.HORIZONTAL).pack(fill=tk.X, pady=12)
        
        # Кнопка запуска
        self.btn = ttk.Button(left, text="▶ Запустить", command=self.run)
        self.btn.pack(fill=tk.X, pady=10)
        
        # Результаты
        result_frame = ttk.LabelFrame(left, text="Результаты", padding=10)
        result_frame.pack(fill=tk.BOTH, expand=True, pady=(10, 0))
        
        self.result_text = tk.Text(result_frame, height=20, width=30, font=("Courier", 9), 
                                   relief=tk.FLAT, bg="#ffffff")
        self.result_text.pack(fill=tk.BOTH, expand=True)
        
        # Правая панель - графики
        self.right = ttk.LabelFrame(main, text="Графики", padding=10)
        self.right.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True)
    
    def run(self):
        try:
            graph_idx = self.graph_var.get()
            graph_name, matrix = self.graphs[graph_idx]
            mode = "classic" if self.mode_var.get() == "Classic" else "cauchy"
            t_start = float(self.t_start.get())
            t_min = float(self.t_min.get())
            alpha = float(self.alpha.get())
            iters = int(self.iters.get())
            
            # Запустить алгоритм
            self.btn.config(state=tk.DISABLED)
            self.result_text.config(state=tk.NORMAL)
            self.result_text.delete(1.0, tk.END)
            self.result_text.insert(tk.END, "⏳ Вычисляю...\n")
            self.root.update()
            
            start_time = time.time()
            result = simulated_annealing(matrix, mode, t_start, t_min, alpha, iters)
            elapsed = time.time() - start_time
            
            # Вывести результаты (короче)
            text = f"Граф: {graph_name}\n"
            text += f"Режим: {self.mode_var.get()}\n"
            text += f"Время: {elapsed:.2f}с\n\n"
            text += f"Лучший путь:\n{result['best_cost']:.2f}\n\n"
            
            if len(result['best_path']) <= 15:
                text += f"Маршрут:\n{result['best_path']}"
            else:
                text += f"Маршрут:\n{result['best_path'][:10]}...\n({len(result['best_path'])} городов)"
            
            self.result_text.delete(1.0, tk.END)
            self.result_text.insert(tk.END, text)
            self.result_text.config(state=tk.DISABLED)
            
            # Построить графики
            self.plot_results(result)
            
            self.btn.config(state=tk.NORMAL)
        except Exception as e:
            self.result_text.config(state=tk.NORMAL)
            self.result_text.delete(1.0, tk.END)
            self.result_text.insert(tk.END, f"❌ Ошибка:\n{e}")
            self.result_text.config(state=tk.DISABLED)
            self.btn.config(state=tk.NORMAL)
    
    def plot_results(self, result):
        # Очистить старые графики
        for widget in self.right.winfo_children():
            widget.destroy()
        
        # Создать фигуру с графиками ВЕРТИКАЛЬНО
        fig = Figure(figsize=(6, 7), dpi=80)
        ax1 = fig.add_subplot(211)
        ax2 = fig.add_subplot(212)
        
        steps = list(range(len(result['history_cost'])))
        
        # График 1: Сходимость
        ax1.plot(steps, result['history_cost'], 'b-', linewidth=1.5)
        ax1.fill_between(steps, result['history_cost'], alpha=0.2)
        ax1.set_xlabel('Шаг охлаждения')
        ax1.set_ylabel('Стоимость пути')
        ax1.set_title('Сходимость алгоритма', fontweight='bold')
        ax1.grid(True, alpha=0.3)
        
        # График 2: Температура
        ax2.plot(steps, result['history_temp'], 'r-', linewidth=1.5)
        ax2.fill_between(steps, result['history_temp'], alpha=0.15, color='red')
        ax2.set_xlabel('Шаг охлаждения')
        ax2.set_ylabel('Температура')
        ax2.set_title('Охлаждение температуры', fontweight='bold')
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