import tkinter as tk
from tkinter import ttk, messagebox, scrolledtext
import threading
import math
import random
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

GRAPHS = {
    0: ("Built-in (6 вершин)", MATRIX_1GRAPH),
    1: ("BERLIN52 (52 города)", BERLIN52),
    2: ("WORLD666 (666 городов)", WORLD666),
}


class TSPSolver:
    """Основной класс алгоритма Simulated Annealing"""
    
    @staticmethod
    def route_cost(matrix, path):
        total = 0
        for i in range(len(path)):
            weight = matrix[path[i]][path[(i + 1) % len(path)]]
            if weight == INF:
                return INF
            total += weight
        return total

    @staticmethod
    def random_path(matrix):
        base_path = list(range(len(matrix)))
        while True:
            path = base_path[:]
            random.shuffle(path)
            if TSPSolver.route_cost(matrix, path) != INF:
                return path

    @staticmethod
    def mutate_path(path):
        new_path = path[:]
        i, j = random.sample(range(len(new_path)), 2)
        new_path[i], new_path[j] = new_path[j], new_path[i]
        return new_path

    @staticmethod
    def next_temperature(mode, t_now, step, t_start, alpha):
        if mode == "classic":
            return t_now * alpha
        if mode == "cauchy":
            return t_start / (1 + step)

    @staticmethod
    def solve(matrix, mode, t_start, t_min, alpha, iters_per_t):
        t = t_start
        current = TSPSolver.random_path(matrix)
        current_cost = TSPSolver.route_cost(matrix, current)
        best_path = current[:]
        best_cost = current_cost
        initial_cost = current_cost
        step = 0
        history_cost = [best_cost]
        history_temp = [t]

        while t > t_min:
            for _ in range(iters_per_t):
                candidate_path = TSPSolver.mutate_path(current)
                candidate_cost = TSPSolver.route_cost(matrix, candidate_path)
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
            t = TSPSolver.next_temperature(mode, t, step, t_start, alpha)
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


class TSPGUI:
    """Графический интерфейс для TSP Solver"""
    
    def __init__(self, root):
        self.root = root
        self.root.title("TSP Solver - Simulated Annealing")
        self.root.geometry("1200x800")
        
        self.result = None
        self.graph_name = ""
        self.graph_size = 0
        
        self.setup_ui()
    
    def setup_ui(self):
        """Создать пользовательский интерфейс"""
        
        # Основной контейнер
        main_frame = ttk.Frame(self.root)
        main_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        # Левая панель - входные параметры
        left_frame = ttk.LabelFrame(main_frame, text="Параметры", padding=10)
        left_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=False, padx=(0, 10))
        
        # Выбор графа
        graph_frame = ttk.LabelFrame(left_frame, text="Граф", padding=5)
        graph_frame.pack(fill=tk.X, pady=5)
        
        self.graph_choice = tk.IntVar(value=1)
        for key, (name, _) in GRAPHS.items():
            ttk.Radiobutton(graph_frame, text=name, variable=self.graph_choice, 
                          value=key).pack(anchor=tk.W)
        
        # Выбор режима охлаждения
        mode_frame = ttk.LabelFrame(left_frame, text="Режим охлаждения", padding=5)
        mode_frame.pack(fill=tk.X, pady=5)
        
        self.mode_choice = tk.StringVar(value="classic")
        ttk.Radiobutton(mode_frame, text="Classic (T = T × ALPHA)", 
                       variable=self.mode_choice, value="classic").pack(anchor=tk.W)
        ttk.Radiobutton(mode_frame, text="Cauchy (T = T_START/(1+step))", 
                       variable=self.mode_choice, value="cauchy").pack(anchor=tk.W)
        
        # Параметры алгоритма
        params_frame = ttk.LabelFrame(left_frame, text="Параметры алгоритма", padding=5)
        params_frame.pack(fill=tk.X, pady=5)
        
        # T_START
        ttk.Label(params_frame, text="T_START:").pack(anchor=tk.W)
        self.t_start_var = tk.StringVar(value="100.0")
        ttk.Entry(params_frame, textvariable=self.t_start_var, width=15).pack(fill=tk.X, pady=2)
        
        # T_MIN
        ttk.Label(params_frame, text="T_MIN:").pack(anchor=tk.W)
        self.t_min_var = tk.StringVar(value="0.089")
        ttk.Entry(params_frame, textvariable=self.t_min_var, width=15).pack(fill=tk.X, pady=2)
        
        # ALPHA (видно только для Classic)
        self.alpha_label = ttk.Label(params_frame, text="ALPHA:")
        self.alpha_label.pack(anchor=tk.W)
        self.alpha_var = tk.StringVar(value="0.95")
        self.alpha_entry = ttk.Entry(params_frame, textvariable=self.alpha_var, width=15)
        self.alpha_entry.pack(fill=tk.X, pady=2)
        
        # Обновить видимость ALPHA при изменении режима
        self.mode_choice.trace("w", self.update_alpha_visibility)
        
        # ITERS_PER_T
        ttk.Label(params_frame, text="ITERS_PER_T:").pack(anchor=tk.W)
        self.iters_var = tk.StringVar(value="30")
        ttk.Entry(params_frame, textvariable=self.iters_var, width=15).pack(fill=tk.X, pady=2)
        
        # Кнопка запуска
        self.run_button = ttk.Button(left_frame, text="▶ Запустить", 
                                     command=self.run_algorithm)
        self.run_button.pack(fill=tk.X, pady=10)
        
        # Правая панель - результаты и графики
        right_frame = ttk.Frame(main_frame)
        right_frame.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True)
        
        # Верхняя часть - результаты
        results_frame = ttk.LabelFrame(right_frame, text="Результаты", padding=5)
        results_frame.pack(fill=tk.BOTH, expand=True, pady=(0, 10))
        
        self.results_text = scrolledtext.ScrolledText(results_frame, height=12, width=50, 
                                                       wrap=tk.WORD, font=("Courier", 9))
        self.results_text.pack(fill=tk.BOTH, expand=True)
        
        # Нижняя часть - графики
        self.canvas_frame = ttk.LabelFrame(right_frame, text="Графики", padding=5)
        self.canvas_frame.pack(fill=tk.BOTH, expand=True)
        
        self.figure = None
        self.canvas = None
        
        self.update_alpha_visibility()
    
    def update_alpha_visibility(self, *args):
        """Показать/скрыть ALPHA в зависимости от режима"""
        if self.mode_choice.get() == "classic":
            self.alpha_label.pack(anchor=tk.W)
            self.alpha_entry.pack(fill=tk.X, pady=2)
        else:
            self.alpha_label.pack_forget()
            self.alpha_entry.pack_forget()
    
    def validate_parameters(self):
        """Проверить корректность параметров"""
        try:
            t_start = float(self.t_start_var.get())
            t_min = float(self.t_min_var.get())
            iters = int(self.iters_var.get())
            
            if t_start <= 0:
                messagebox.showerror("Ошибка", "T_START должна быть положительной")
                return False
            if not (0 < t_min < t_start):
                messagebox.showerror("Ошибка", f"T_MIN должна быть между 0 и {t_start}")
                return False
            if iters <= 0:
                messagebox.showerror("Ошибка", "ITERS_PER_T должна быть положительной")
                return False
            
            if self.mode_choice.get() == "classic":
                alpha = float(self.alpha_var.get())
                if not (0 < alpha < 1):
                    messagebox.showerror("Ошибка", "ALPHA должна быть между 0 и 1")
                    return False
            
            return True
        except ValueError as e:
            messagebox.showerror("Ошибка", f"Неверный формат параметра: {e}")
            return False
    
    def run_algorithm(self):
        """Запустить алгоритм в отдельном потоке"""
        if not self.validate_parameters():
            return
        
        self.run_button.config(state=tk.DISABLED)
        self.results_text.config(state=tk.NORMAL)
        self.results_text.delete(1.0, tk.END)
        self.results_text.insert(tk.END, "⏳ Выполняется вычисление...\n")
        self.results_text.config(state=tk.DISABLED)
        
        thread = threading.Thread(target=self._solve_thread)
        thread.daemon = True
        thread.start()
    
    def _solve_thread(self):
        """Поток для выполнения алгоритма"""
        try:
            graph_choice = self.graph_choice.get()
            self.graph_name = GRAPHS[graph_choice][0]
            matrix = GRAPHS[graph_choice][1]
            self.graph_size = len(matrix)
            
            mode = self.mode_choice.get()
            t_start = float(self.t_start_var.get())
            t_min = float(self.t_min_var.get())
            iters_per_t = int(self.iters_var.get())
            alpha = float(self.alpha_var.get()) if mode == "classic" else None
            
            self.result = TSPSolver.solve(matrix, mode, t_start, t_min, alpha, iters_per_t)
            
            # Обновить UI в главном потоке
            self.root.after(0, self.display_results)
            
        except Exception as e:
            self.root.after(0, lambda: messagebox.showerror("Ошибка", str(e)))
        finally:
            self.root.after(0, lambda: self.run_button.config(state=tk.NORMAL))
    
    def display_results(self):
        """Отобразить результаты"""
        if not self.result:
            return
        
        # Текстовые результаты
        self.results_text.config(state=tk.NORMAL)
        self.results_text.delete(1.0, tk.END)
        
        output = "=" * 50 + "\n"
        output += "РЕЗУЛЬТАТЫ\n"
        output += "=" * 50 + "\n\n"
        
        output += "[ПАРАМЕТРЫ]\n"
        output += f"  График: {self.graph_name} ({self.graph_size} вершин)\n"
        output += f"  Режим: {self.result['mode'].upper()}\n"
        output += f"  T_START: {self.t_start_var.get()}\n"
        output += f"  T_MIN: {self.t_min_var.get()}\n"
        if self.result['mode'] == "classic":
            output += f"  ALPHA: {self.alpha_var.get()}\n"
        output += f"  ITERS_PER_T: {self.iters_var.get()}\n"
        
        output += "\n[РЕЗУЛЬТАТЫ]\n"
        initial = self.result['initial_cost']
        best = self.result['best_cost']
        improvement = ((initial - best) / initial * 100) if initial > 0 else 0
        
        output += f"  Начальная длина: {initial:.2f}\n"
        output += f"  Лучшая длина: {best:.2f}\n"
        output += f"  Улучшение: {improvement:.1f}%\n"
        output += f"  Шагов охлаждения: {self.result['steps']}\n"
        
        if self.graph_size <= 10:
            output += f"\n  Путь: {self.result['best_path']}\n"
        else:
            output += f"\n  Путь (первые 10): {self.result['best_path'][:10]} ...\n"
        
        self.results_text.insert(tk.END, output)
        self.results_text.config(state=tk.DISABLED)
        
        # Графики
        self.plot_results()
    
    def plot_results(self):
        """Построить графики"""
        # Удалить старый canvas если существует
        if self.canvas:
            self.canvas.get_tk_widget().destroy()
        
        # Создать новую фигуру
        self.figure = Figure(figsize=(10, 4), dpi=100)
        ax1 = self.figure.add_subplot(121)
        ax2 = self.figure.add_subplot(122)
        
        # График 1: Сходимость
        steps = list(range(len(self.result['history_cost'])))
        ax1.plot(steps, self.result['history_cost'], linewidth=1.5, color='blue')
        ax1.set_xlabel('Шаг охлаждения')
        ax1.set_ylabel('Лучшая длина пути')
        ax1.set_title('Сходимость алгоритма')
        ax1.grid(True, alpha=0.3)
        
        # График 2: Температура
        ax2.plot(steps, self.result['history_temp'], linewidth=1.5, color='red')
        ax2.set_xlabel('Шаг охлаждения')
        ax2.set_ylabel('Температура')
        ax2.set_title('Охлаждение температуры')
        ax2.grid(True, alpha=0.3)
        
        self.figure.tight_layout()
        
        # Встроить в GUI
        self.canvas = FigureCanvasTkAgg(self.figure, master=self.canvas_frame)
        self.canvas.draw()
        self.canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)


def main():
    root = tk.Tk()
    gui = TSPGUI(root)
    root.mainloop()


if __name__ == "__main__":
    main()
