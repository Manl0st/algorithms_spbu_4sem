import math
import random
import time
import tkinter as tk
from tkinter import ttk
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure
from matplotlib.ticker import FuncFormatter, MaxNLocator
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

def mutate_path_cauchy(path):
    new_path = path[:]
    n = len(new_path)
    if n < 2:
        return new_path

    i = random.randrange(n)
    u = random.random()
    cauchy_value = math.tan(math.pi * (u - 0.5))
    shift = int(abs(cauchy_value)) % (n - 1) + 1
    j = (i + shift) % n

    new_path[i], new_path[j] = new_path[j], new_path[i]
    return new_path

def next_temperature(mode, t_now, step, t_start, alpha):
    if mode == "Базовый":
        return t_now * alpha
    if mode == "Отжиг Коши":
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
            if mode == "Отжиг Коши":
                candidate_path = mutate_path_cauchy(current)
            else:
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
        self.root.geometry("1320x700")
        self.root.configure(bg="#f0f0f0")
        
        self.graphs = {
            0: ("Ориентированный", MATRIX_1GRAPH),
            1: ("BERLIN52", BERLIN52),
            2: ("WORLD666", WORLD666),
        }
        self.graph_by_name = {name: matrix for _, (name, matrix) in self.graphs.items()}
        
        self.setup_ui()
    
    def setup_ui(self):
        main = ttk.Frame(self.root, padding=12)
        main.pack(fill=tk.BOTH, expand=True)
        main.columnconfigure(0, weight=1, minsize=300)
        main.columnconfigure(1, weight=5, minsize=900)
        main.rowconfigure(0, weight=1)

        left = ttk.Frame(main)
        left.grid(row=0, column=0, sticky="nsew", padx=(0, 10))

        params_frame = ttk.LabelFrame(left, text="Параметры", padding=8)
        params_frame.pack(fill=tk.X)
        params_frame.columnconfigure(1, weight=1)

        ttk.Label(params_frame, text="Граф:").grid(row=0, column=0, sticky="w", padx=(0, 8), pady=4)
        graph_names = [name for _, (name, _) in self.graphs.items()]
        self.graph_var = tk.StringVar(value=graph_names[1])
        graph_combo = ttk.Combobox(
            params_frame,
            textvariable=self.graph_var,
            values=graph_names,
            state="readonly",
            width=16,
        )
        graph_combo.grid(row=0, column=1, sticky="ew", pady=4)

        ttk.Label(params_frame, text="Модификация:").grid(row=1, column=0, sticky="w", padx=(0, 8), pady=4)
        self.mode_var = tk.StringVar(value="Базовый")
        mode_combo = ttk.Combobox(
            params_frame,
            textvariable=self.mode_var,
            values=["Базовый", "Отжиг Коши"],
            state="readonly",
            width=16,
        )
        mode_combo.grid(row=1, column=1, sticky="ew", pady=4)

        ttk.Label(params_frame, text="T начальная:").grid(row=2, column=0, sticky="w", padx=(0, 8), pady=4)
        self.t_start = ttk.Entry(params_frame, width=16)
        self.t_start.insert(0, "100.0")
        self.t_start.grid(row=2, column=1, sticky="ew", pady=4)

        ttk.Label(params_frame, text="T минимальная:").grid(row=3, column=0, sticky="w", padx=(0, 8), pady=4)
        self.t_min = ttk.Entry(params_frame, width=16)
        self.t_min.insert(0, "0.089")
        self.t_min.grid(row=3, column=1, sticky="ew", pady=4)

        ttk.Label(params_frame, text="коэфф. a:").grid(row=4, column=0, sticky="w", padx=(0, 8), pady=4)
        self.alpha = ttk.Entry(params_frame, width=16)
        self.alpha.insert(0, "0.95")
        self.alpha.grid(row=4, column=1, sticky="ew", pady=4)

        ttk.Label(params_frame, text="итераций на T:").grid(row=5, column=0, sticky="w", padx=(0, 8), pady=4)
        self.iters = ttk.Entry(params_frame, width=16)
        self.iters.insert(0, "30")
        self.iters.grid(row=5, column=1, sticky="ew", pady=4)

        self.btn = ttk.Button(params_frame, text="▶ Запустить", command=self.run)
        self.btn.grid(row=6, column=0, columnspan=2, sticky="ew", pady=(8, 2))

        result_frame = ttk.LabelFrame(left, text="Результаты", padding=10)
        result_frame.pack(fill=tk.BOTH, expand=True, pady=(10, 0))

        self.result_text = tk.Text(
            result_frame,
            height=20,
            width=30,
            font=("Courier", 10),
            relief=tk.FLAT,
            bg="#ffffff",
            wrap=tk.WORD,
        )
        self.result_text.pack(fill=tk.BOTH, expand=True)

        self.right = ttk.LabelFrame(main, text="Графики", padding=10)
        self.right.grid(row=0, column=1, sticky="nsew")
    
    def run(self):
        try:
            graph_name = self.graph_var.get()
            matrix = self.graph_by_name[graph_name]
            mode = "Базовый" if self.mode_var.get() == "Базовый" else "Отжиг Коши"
            t_start = float(self.t_start.get())
            t_min = float(self.t_min.get())
            alpha = float(self.alpha.get())
            iters = int(self.iters.get())
            
            # Запустить алгоритм
            self.btn.config(state=tk.DISABLED)
            self.result_text.config(state=tk.NORMAL)
            self.result_text.delete(1.0, tk.END)
            self.result_text.insert(tk.END, "Подсчет...\n")
            self.root.update()
            
            start_time = time.time()
            result = simulated_annealing(matrix, mode, t_start, t_min, alpha, iters)
            elapsed = time.time() - start_time
            
            text = f"Длина лучшего пути: {result['best_cost']:.2f}\n"
            text += f"Время работы: {elapsed:.2f} с\n\n"
            text += f"Лучший путь:\n{result['best_path']}"
            
            self.result_text.delete(1.0, tk.END)
            self.result_text.insert(tk.END, text)
            self.result_text.config(state=tk.DISABLED)
            
            # Построить графики
            self.plot_results(result)
            
            self.btn.config(state=tk.NORMAL)
        except Exception as e:
            self.result_text.config(state=tk.NORMAL)
            self.result_text.delete(1.0, tk.END)
            self.result_text.insert(tk.END, f"Ошибка:\n{e}")
            self.result_text.config(state=tk.DISABLED)
            self.btn.config(state=tk.NORMAL)
    
    def plot_results(self, result):
        # Очистить старые графики
        for widget in self.right.winfo_children():
            widget.destroy()
        
        # Создать фигуру с графиками ВЕРТИКАЛЬНО
        fig = Figure(figsize=(8.5, 8.0), dpi=90)
        ax1 = fig.add_subplot(211)
        ax2 = fig.add_subplot(212)
        
        steps = list(range(len(result['history_cost'])))

        def build_ticks(max_value, target_ticks=8):
            if max_value <= 0:
                return [0, 1], 1
            raw_step = max_value / target_ticks
            power = 10 ** math.floor(math.log10(raw_step))
            ratio = raw_step / power
            if ratio <= 1:
                step = power
            elif ratio <= 2:
                step = 2 * power
            elif ratio <= 5:
                step = 5 * power
            else:
                step = 10 * power
            top = math.ceil(max_value / step) * step
            ticks = [i * step for i in range(int(top / step) + 1)]
            return ticks, step
        
        # График 1: Сходимость
        ax1.plot(steps, result['history_cost'], 'b-', linewidth=1.5)
        ax1.set_xlim(left=0)
        cost_ticks, cost_step = build_ticks(max(result['history_cost']))
        ax1.set_ylim(0, cost_ticks[-1])
        ax1.set_yticks(cost_ticks)
        ax1.set_xlabel('Шаг охлаждения')
        ax1.set_ylabel('Длина пути')
        ax1.set_title('Сходимость алгоритма', fontweight='bold')
        ax1.xaxis.set_major_locator(MaxNLocator(nbins=8, integer=True))
        if cost_step < 1:
            digits = min(6, max(0, int(-math.floor(math.log10(cost_step)))))
            ax1.yaxis.set_major_formatter(FuncFormatter(lambda x, _: f"{x:.{digits}f}"))
        else:
            ax1.yaxis.set_major_formatter(FuncFormatter(lambda x, _: f"{x:,.0f}".replace(",", " ")))
        ax1.minorticks_on()
        ax1.grid(True, which='major', alpha=0.35)
        ax1.grid(True, which='minor', alpha=0.15, linestyle=':')
        
        # График 2: Температура
        ax2.plot(steps, result['history_temp'], 'r-', linewidth=1.5)
        ax2.set_xlim(left=0)
        temp_ticks, temp_step = build_ticks(max(result['history_temp']))
        ax2.set_ylim(0, temp_ticks[-1])
        ax2.set_yticks(temp_ticks)
        ax2.set_xlabel('Шаг охлаждения')
        ax2.set_ylabel('Температура')
        ax2.set_title('Охлаждение температуры', fontweight='bold')
        ax2.xaxis.set_major_locator(MaxNLocator(nbins=8, integer=True))
        if temp_step < 1:
            digits = min(6, max(0, int(-math.floor(math.log10(temp_step)))))
            ax2.yaxis.set_major_formatter(FuncFormatter(lambda y, _: f"{y:.{digits}f}"))
        else:
            ax2.yaxis.set_major_formatter(FuncFormatter(lambda y, _: f"{y:g}"))
        ax2.minorticks_on()
        ax2.grid(True, which='major', alpha=0.35)
        ax2.grid(True, which='minor', alpha=0.15, linestyle=':')
        
        fig.tight_layout()
        
        # Встроить в GUI
        canvas = FigureCanvasTkAgg(fig, master=self.right)
        canvas.draw()
        canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)

if __name__ == "__main__":
    root = tk.Tk()
    app = TSPGui(root)
    root.mainloop()
