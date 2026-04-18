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
    [0, 3, INF, INF, 1, INF],
    [3, 0, 8, INF, INF, 3],
    [INF, 3, 0, 1, INF, 3],
    [INF, INF, 8, 0, 3, INF],
    [3, INF, INF, 1, 0, INF],
    [3, INF, 1, 5, 4, 0],
]


def route_cost(matrix, path):
    total = 0
    n = len(path)
    for i in range(n):
        a = path[i]
        b = path[(i + 1) % n]
        w = matrix[a][b]
        if w == INF:
            return INF
        total += w
    return total


def build_heuristic(matrix):
    n = len(matrix)
    h = [[0.0] * n for _ in range(n)]
    for i in range(n):
        for j in range(n):
            d = matrix[i][j]
            if i != j and d != INF and d > 0:
                h[i][j] = 1.0 / d
    return h


def build_neighbors(matrix):
    n = len(matrix)
    neighbors = []
    for i in range(n):
        row = [j for j in range(n) if i != j and matrix[i][j] != INF]
        neighbors.append(row)
    return neighbors


def choose_next_city(current, unvisited, pheromone, heuristic, alpha, beta, neighbors):
    candidates = [j for j in neighbors[current] if j in unvisited]
    if not candidates:
        return None

    weights = []
    for j in candidates:
        tau = pheromone[current][j] ** alpha
        eta = heuristic[current][j] ** beta
        weights.append(tau * eta)

    total = sum(weights)
    if total <= 0:
        return random.choice(candidates)

    r = random.random() * total
    acc = 0.0
    for city, w in zip(candidates, weights):
        acc += w
        if acc >= r:
            return city
    return candidates[-1]


def build_ant_tour(matrix, pheromone, heuristic, neighbors, alpha, beta):
    n = len(matrix)
    start = random.randrange(n)
    path = [start]
    unvisited = set(range(n))
    unvisited.remove(start)
    total = 0.0

    while unvisited:
        current = path[-1]
        nxt = choose_next_city(current, unvisited, pheromone, heuristic, alpha, beta, neighbors)
        if nxt is None:
            return None, INF
        w = matrix[current][nxt]
        if w == INF:
            return None, INF
        total += w
        path.append(nxt)
        unvisited.remove(nxt)

    back = matrix[path[-1]][start]
    if back == INF:
        return None, INF
    total += back
    return path, total


def tour_edges(path):
    n = len(path)
    for i in range(n):
        yield path[i], path[(i + 1) % n]


def run_aco(matrix, mode, ants_count, iterations, alpha, beta, rho, q, elite_weight):
    n = len(matrix)
    pheromone = [[1.0] * n for _ in range(n)]
    for i in range(n):
        pheromone[i][i] = 0.0

    heuristic = build_heuristic(matrix)
    neighbors = build_neighbors(matrix)

    best_path = None
    best_cost = INF
    best_history = []
    avg_history = []

    for _ in range(iterations):
        tours = []
        sum_cost = 0.0
        count = 0

        for _ in range(ants_count):
            path, cost = build_ant_tour(matrix, pheromone, heuristic, neighbors, alpha, beta)
            if path is None or cost == INF:
                continue
            tours.append((path, cost))
            sum_cost += cost
            count += 1
            if cost < best_cost:
                best_cost = cost
                best_path = path[:]

        evap = 1.0 - rho
        for i in range(n):
            row = pheromone[i]
            for j in range(n):
                row[j] *= evap
                if row[j] < 1e-12:
                    row[j] = 1e-12

        for path, cost in tours:
            delta = q / cost
            for a, b in tour_edges(path):
                pheromone[a][b] += delta
                pheromone[b][a] += delta

        if mode == "elite" and best_path is not None:
            delta = elite_weight * (q / best_cost)
            for a, b in tour_edges(best_path):
                pheromone[a][b] += delta
                pheromone[b][a] += delta

        if best_cost == INF:
            best_history.append(0.0)
            avg_history.append(0.0)
        else:
            best_history.append(best_cost)
            if count > 0:
                avg_history.append(sum_cost / count)
            else:
                avg_history.append(best_cost)

    return {
        "best_path": best_path,
        "best_cost": best_cost,
        "best_history": best_history,
        "avg_history": avg_history,
    }


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


class ACOGui:
    def __init__(self, root):
        self.root = root
        self.root.title("TSP ACO")
        self.root.geometry("1320x700")
        self.root.configure(bg="#f0f0f0")
        self.graphs = {
            "Ориентированный": MATRIX_1GRAPH,
            "BERLIN52": BERLIN52,
            "WORLD666": WORLD666,
        }
        self.setup_ui()

    def setup_ui(self):
        main = ttk.Frame(self.root, padding=12)
        main.pack(fill=tk.BOTH, expand=True)
        main.columnconfigure(0, weight=1, minsize=320)
        main.columnconfigure(1, weight=5, minsize=900)
        main.rowconfigure(0, weight=1)
        left = ttk.Frame(main)
        left.grid(row=0, column=0, sticky="nsew", padx=(0, 10))
        params = ttk.LabelFrame(left, text="Параметры", padding=8)
        params.pack(fill=tk.X)
        params.columnconfigure(1, weight=1)
        ttk.Label(params, text="Граф:").grid(row=0, column=0, sticky="w", padx=(0, 8), pady=4)
        self.graph_var = tk.StringVar(value="BERLIN52")
        ttk.Combobox(
            params,
            textvariable=self.graph_var,
            values=list(self.graphs.keys()),
            state="readonly",
            width=16,
        ).grid(row=0, column=1, sticky="ew", pady=4)
        ttk.Label(params, text="Модификация:").grid(row=1, column=0, sticky="w", padx=(0, 8), pady=4)
        self.mode_var = tk.StringVar(value="Базовый")
        ttk.Combobox(
            params,
            textvariable=self.mode_var,
            values=["Базовый", "Элитные муравьи"],
            state="readonly",
            width=16,
        ).grid(row=1, column=1, sticky="ew", pady=4)
        ttk.Label(params, text="Муравьи:").grid(row=2, column=0, sticky="w", padx=(0, 8), pady=4)
        self.ants_var = ttk.Entry(params, width=16)
        self.ants_var.insert(0, "12")
        self.ants_var.grid(row=2, column=1, sticky="ew", pady=4)
        ttk.Label(params, text="Итерации:").grid(row=3, column=0, sticky="w", padx=(0, 8), pady=4)
        self.iter_var = ttk.Entry(params, width=16)
        self.iter_var.insert(0, "40")
        self.iter_var.grid(row=3, column=1, sticky="ew", pady=4)
        ttk.Label(params, text="ALPHA:").grid(row=4, column=0, sticky="w", padx=(0, 8), pady=4)
        self.alpha_var = ttk.Entry(params, width=16)
        self.alpha_var.insert(0, "1.0")
        self.alpha_var.grid(row=4, column=1, sticky="ew", pady=4)
        ttk.Label(params, text="BETA:").grid(row=5, column=0, sticky="w", padx=(0, 8), pady=4)
        self.beta_var = ttk.Entry(params, width=16)
        self.beta_var.insert(0, "3.0")
        self.beta_var.grid(row=5, column=1, sticky="ew", pady=4)
        ttk.Label(params, text="RHO:").grid(row=6, column=0, sticky="w", padx=(0, 8), pady=4)
        self.rho_var = ttk.Entry(params, width=16)
        self.rho_var.insert(0, "0.5")
        self.rho_var.grid(row=6, column=1, sticky="ew", pady=4)
        ttk.Label(params, text="Q:").grid(row=7, column=0, sticky="w", padx=(0, 8), pady=4)
        self.q_var = ttk.Entry(params, width=16)
        self.q_var.insert(0, "100.0")
        self.q_var.grid(row=7, column=1, sticky="ew", pady=4)
        ttk.Label(params, text="Elite weight:").grid(row=8, column=0, sticky="w", padx=(0, 8), pady=4)
        self.elite_var = ttk.Entry(params, width=16)
        self.elite_var.insert(0, "2.0")
        self.elite_var.grid(row=8, column=1, sticky="ew", pady=4)
        self.run_btn = ttk.Button(params, text="Запустить", command=self.run)
        self.run_btn.grid(row=9, column=0, columnspan=2, sticky="ew", pady=(8, 2))
        result = ttk.LabelFrame(left, text="Результаты", padding=10)
        result.pack(fill=tk.BOTH, expand=True, pady=(10, 0))
        self.result_text = tk.Text(result, height=20, width=30, font=("Courier", 10), relief=tk.FLAT, bg="#ffffff", wrap=tk.WORD)
        self.result_text.pack(fill=tk.BOTH, expand=True)
        self.result_text.config(state=tk.DISABLED)
        self.right = ttk.LabelFrame(main, text="Графики", padding=10)
        self.right.grid(row=0, column=1, sticky="nsew")

    def run(self):
        try:
            matrix = self.graphs[self.graph_var.get()]
            mode = "elite" if self.mode_var.get() == "Elite" else "base"
            ants = int(self.ants_var.get())
            iterations = int(self.iter_var.get())
            alpha = float(self.alpha_var.get())
            beta = float(self.beta_var.get())
            rho = float(self.rho_var.get())
            q = float(self.q_var.get())
            elite_weight = float(self.elite_var.get())
            self.run_btn.config(state=tk.DISABLED)
            self.result_text.config(state=tk.NORMAL)
            self.result_text.delete(1.0, tk.END)
            self.result_text.insert(tk.END, "Подсчет...\n")
            self.result_text.config(state=tk.DISABLED)
            self.root.update()
            t0 = time.time()
            out = run_aco(matrix, mode, ants, iterations, alpha, beta, rho, q, elite_weight)
            dt = time.time() - t0
            if out["best_path"] is None or out["best_cost"] == INF:
                text = "Путь не найден\n"
                text += f"Время: {dt:.2f} с"
            else:
                text = f"Длина лучшего пути: {out['best_cost']:.2f}\n"
                text += f"Время работы: {dt:.2f} с\n\n"
                text += f"Лучший путь:\n{out['best_path']}"
            self.result_text.config(state=tk.NORMAL)
            self.result_text.delete(1.0, tk.END)
            self.result_text.insert(tk.END, text)
            self.result_text.config(state=tk.DISABLED)
            self.plot_results(out)
            self.run_btn.config(state=tk.NORMAL)
        except Exception as e:
            self.result_text.config(state=tk.NORMAL)
            self.result_text.delete(1.0, tk.END)
            self.result_text.insert(tk.END, f"Ошибка:\n{e}")
            self.result_text.config(state=tk.DISABLED)
            self.run_btn.config(state=tk.NORMAL)

    def plot_results(self, out):
        for w in self.right.winfo_children():
            w.destroy()
        fig = Figure(figsize=(8.5, 8.0), dpi=90)
        ax1 = fig.add_subplot(211)
        ax2 = fig.add_subplot(212)
        x = list(range(1, len(out["best_history"]) + 1))
        best = out["best_history"]
        avg = out["avg_history"]
        ax1.plot(x, best, "b-", linewidth=1.6)
        ax1.set_xlim(left=1)
        max_best = max(best) if best else 1
        ticks_best, step_best = build_ticks(max_best)
        ax1.set_ylim(0, ticks_best[-1])
        ax1.set_yticks(ticks_best)
        ax1.set_xlabel("Итерация")
        ax1.set_ylabel("Лучшая длина")
        ax1.set_title("Сходимость ACO", fontweight="bold")
        ax1.xaxis.set_major_locator(MaxNLocator(nbins=8, integer=True))
        if step_best < 1:
            d = min(6, max(0, int(-math.floor(math.log10(step_best)))))
            ax1.yaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:.{d}f}"))
        else:
            ax1.yaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:,.0f}".replace(",", " ")))
        ax1.minorticks_on()
        ax1.grid(True, which="major", alpha=0.35)
        ax1.grid(True, which="minor", alpha=0.15, linestyle=":")
        ax2.plot(x, avg, color="darkorange", linewidth=1.6)
        ax2.set_xlim(left=1)
        max_avg = max(avg) if avg else 1
        ticks_avg, step_avg = build_ticks(max_avg)
        ax2.set_ylim(0, ticks_avg[-1])
        ax2.set_yticks(ticks_avg)
        ax2.set_xlabel("Итерация")
        ax2.set_ylabel("Средняя длина")
        ax2.set_title("Средняя длина туров", fontweight="bold")
        ax2.xaxis.set_major_locator(MaxNLocator(nbins=8, integer=True))
        if step_avg < 1:
            d = min(6, max(0, int(-math.floor(math.log10(step_avg)))))
            ax2.yaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:.{d}f}"))
        else:
            ax2.yaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:,.0f}".replace(",", " ")))
        ax2.minorticks_on()
        ax2.grid(True, which="major", alpha=0.35)
        ax2.grid(True, which="minor", alpha=0.15, linestyle=":")
        fig.tight_layout()
        canvas = FigureCanvasTkAgg(fig, master=self.right)
        canvas.draw()
        canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)


if __name__ == "__main__":
    root = tk.Tk()
    ACOGui(root)
    root.mainloop()
