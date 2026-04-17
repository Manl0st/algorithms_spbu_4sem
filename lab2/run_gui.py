#!/usr/bin/env python3
"""
Quick start script for TSP Solver GUI
Запускает графический интерфейс приложения
"""

import sys
import tkinter as tk

try:
    from gui import TSPGUI
    
    root = tk.Tk()
    app = TSPGUI(root)
    root.mainloop()
    
except ImportError as e:
    print(f"❌ Ошибка: {e}")
    print("Убедитесь, что установлены зависимости:")
    print("  pip install matplotlib")
    sys.exit(1)
except Exception as e:
    print(f"❌ Неожиданная ошибка: {e}")
    sys.exit(1)
