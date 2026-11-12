import sys
from PyQt6.QtWidgets import QApplication

from core.world import World
from ui.map_canvas import MapViewer


if __name__ == "__main__":
    app = QApplication(sys.argv)

    # Создаём мир (seed=None для случайной карты при каждом запуске)
    world = World(width=200, height=150, seed=None)

    # Создаём и показываем окно визуализации
    viewer = MapViewer(world)
    viewer.show()

    sys.exit(app.exec())