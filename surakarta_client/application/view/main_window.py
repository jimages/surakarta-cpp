from PyQt5.QtWidgets import QWidget, QDesktopWidget, QGridLayout


SCREEN_WIDTH = 800
SCREEN_HEIGHT = 500


class MainWindow(QWidget):

    def __init__(self, parent=None):
        super(MainWindow, self).__init__(parent)
        self.setWindowTitle('qyz_surakarta')
        self.resize(SCREEN_WIDTH, SCREEN_HEIGHT)
        center_x, center_y = self._get_screen_center()
        self.move(center_x, center_y)

    @staticmethod
    def _get_screen_center():
        screen = QDesktopWidget().screenGeometry()
        return (screen.width() - SCREEN_WIDTH) / 2, (screen.height() - SCREEN_HEIGHT) / 2
