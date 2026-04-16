import math
import tkinter as tk
from pathlib import Path
from tkinter import messagebox, ttk

from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure
import numpy as np


INF = float("inf")


def load_h_txt(path):
    env = {"__builtins__": {}, "float": float}
    local = {}
    exec(path.read_text(encoding="utf-8"), env, local)
    matrix = local.get("matrix") or env.get("matrix")
    if matrix is None:
        raise ValueError("В h.txt нет переменной matrix")
    return np.array(matrix, dtype=float), True


def load_stp(path, directed=False):
    n = None
    edges = []

    with path.open("r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            parts = line.split()
            if not parts:
                continue
            if parts[0] == "Nodes":
                n = int(parts[1])
            elif parts[0] in ("E", "A") and len(parts) >= 4:
                u = int(parts[1]) - 1
                v = int(parts[2]) - 1
                w = float(parts[3])
                edges.append((u, v, w))

    if n is None:
        raise ValueError(f"Не удалось прочитать Nodes в {path.name}")

    D = np.full((n, n), INF, dtype=float)
    np.fill_diagonal(D, 0.0)
    for u, v, w in edges:
        D[u, v] = w
        if not directed:
            D[v, u] = w

    return D, directed


def L(route, D):
    s = 0.0
    for i in range(len(route) - 1):
        w = D[route[i], route[i + 1]]
        if math.isinf(w):
            return INF
        s += w
    return s


def random_valid_route(D, tries=5000):
    n = D.shape[0]
    for _ in range(tries):
        body = np.random.permutation(np.arange(1, n))
        route = np.concatenate(([0], body, [0]))
        if math.isfinite(L(route, D)):
            return route
    raise ValueError("Не удалось найти допустимый стартовый маршрут")


def transpose(route):
    tmp = route.copy()
    i, j = np.random.choice(np.arange(1, len(tmp) - 1), size=2, replace=False)
    tmp[i], tmp[j] = tmp[j], tmp[i]
    return tmp


def SA(D, T0=1000.0, alpha=0.995, iterations=20000, cauchy=False):
    conf_i = random_valid_route(D)
    L_i = L(conf_i, D)

    best_conf = conf_i.copy()
    best_val = L_i
    history = [best_val]

    for k in range(1, iterations + 1):
        T = T0 / (1 + k) if cauchy else T0 * (alpha ** k)
        if T <= 1e-12:
            history.append(best_val)
            continue

        conf_t = transpose(conf_i)
        L_t = L(conf_t, D)

        if math.isfinite(L_t):
            dE = L_t - L_i
            if dE <= 0 or np.random.rand() < np.exp(-dE / T):
                conf_i = conf_t
                L_i = L_t
                if L_i < best_val:
                    best_val = L_i
                    best_conf = conf_i.copy()

        history.append(best_val)

    return best_conf, best_val, history


def short_route_text(route, max_nodes=12):
    nodes = [str(int(x) + 1) for x in route]
    if len(nodes) <= max_nodes:
        return " -> ".join(nodes)
    k = max_nodes // 2
    return " -> ".join(nodes[:k]) + " -> ... -> " + " -> ".join(nodes[-k:])


class App:
    def __init__(self, root):
        self.root = root
        self.root.title("Алгоритм отжига")

        self.base = Path(__file__).resolve().parent
        self.graphs = {
            "h.txt (ориентированный)": ("h.txt", True),
            "berlin52.stp (неориентированный)": ("berlin52.stp", False),
            "world666.stp (неориентированный)": ("world666.stp", False),
        }

        self.graph_var = tk.StringVar(value="h.txt (ориентированный)")
        self.mode_var = tk.StringVar(value="Без модификации")
        self.t0_var = tk.DoubleVar(value=1500.0)
        self.alpha_var = tk.DoubleVar(value=0.995)
        self.iters_var = tk.IntVar(value=10000)
        self.res_var = tk.StringVar(value="")

        frm = ttk.Frame(root, padding=10)
        frm.pack(fill="both", expand=True)

        ttk.Label(frm, text="Граф").grid(row=0, column=0, sticky="w")
        ttk.Combobox(
            frm,
            values=list(self.graphs.keys()),
            textvariable=self.graph_var,
            state="readonly",
            width=36,
        ).grid(row=0, column=1, sticky="ew", pady=2)

        ttk.Label(frm, text="Режим").grid(row=1, column=0, sticky="w")
        ttk.Combobox(
            frm,
            values=["Без модификации", "С модификацией (Коши)"],
            textvariable=self.mode_var,
            state="readonly",
            width=36,
        ).grid(row=1, column=1, sticky="ew", pady=2)

        ttk.Label(frm, text="T0").grid(row=2, column=0, sticky="w")
        ttk.Entry(frm, textvariable=self.t0_var, width=12).grid(row=2, column=1, sticky="w", pady=2)

        ttk.Label(frm, text="alpha").grid(row=3, column=0, sticky="w")
        ttk.Entry(frm, textvariable=self.alpha_var, width=12).grid(row=3, column=1, sticky="w", pady=2)

        ttk.Label(frm, text="Итерации").grid(row=4, column=0, sticky="w")
        ttk.Entry(frm, textvariable=self.iters_var, width=12).grid(row=4, column=1, sticky="w", pady=2)

        ttk.Button(frm, text="Запустить", command=self.run).grid(row=5, column=0, columnspan=2, sticky="ew", pady=6)
        ttk.Label(frm, textvariable=self.res_var, justify="left", wraplength=900).grid(
            row=6, column=0, columnspan=2, sticky="w"
        )

        self.figure = Figure(figsize=(7, 3), dpi=100)
        self.ax = self.figure.add_subplot(111)
        self.ax.set_title("Сходимость")
        self.ax.set_xlabel("Итерация")
        self.ax.set_ylabel("Лучшая длина")

        self.canvas = FigureCanvasTkAgg(self.figure, master=frm)
        self.canvas.get_tk_widget().grid(row=7, column=0, columnspan=2, sticky="nsew", pady=(8, 0))
        self.canvas.draw()

        frm.columnconfigure(1, weight=1)
        frm.rowconfigure(7, weight=1)

    def run(self):
        try:
            graph_name = self.graph_var.get()
            file_name, directed = self.graphs[graph_name]
            path = self.base / file_name

            if file_name.endswith(".txt"):
                D, directed = load_h_txt(path)
            else:
                D, directed = load_stp(path, directed=directed)

            T0 = float(self.t0_var.get())
            alpha = float(self.alpha_var.get())
            iterations = int(self.iters_var.get())
            cauchy = self.mode_var.get().startswith("С модификацией")

            if T0 <= 0 or not (0 < alpha < 1) or iterations <= 0:
                raise ValueError("Нужны параметры: T0 > 0, 0 < alpha < 1, iterations > 0")

            best_conf, best_val, history = SA(D, T0, alpha, iterations, cauchy=cauchy)

            gtype = "ориентированный" if directed else "неориентированный"
            self.res_var.set(
                f"Граф: {file_name} ({gtype})\n"
                f"Длина пути: {best_val:.2f}\n"
                f"Путь: {short_route_text(best_conf)}"
            )

            self.ax.clear()
            self.ax.plot(history, lw=1.2)
            self.ax.set_xlabel("Итерация")
            self.ax.set_ylabel("Лучшая длина")
            self.ax.set_title("Отжиг Коши" if cauchy else "Обычный отжиг")
            self.figure.tight_layout()
            self.canvas.draw()

        except Exception as e:
            messagebox.showerror("Ошибка", str(e))


if __name__ == "__main__":
    root = tk.Tk()
    App(root)
    root.mainloop()
