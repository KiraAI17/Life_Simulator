import sys
from PyQt6.QtWidgets import QApplication

from core.world import World
from ui.main_window import MainWindow


if __name__ == "__main__":
    app = QApplication(sys.argv)
    world = World(width=500, height=350, seed=None)
    window = MainWindow(world)
    window.show()
    sys.exit(app.exec())