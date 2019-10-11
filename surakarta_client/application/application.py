from PyQt5.QtWidgets import QApplication
from application.controller import game_controller


def launch():
    import sys
    app = QApplication(sys.argv)
    controller = game_controller.GameController()
    controller.app_launch()
    sys.exit(app.exec_())
