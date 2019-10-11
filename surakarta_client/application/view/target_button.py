from PyQt5.QtWidgets import QPushButton


class TargetButton(QPushButton):

    def __init__(self, *__args):
        super().__init__(*__args)
        self.x = None
        self.y = None
        self.size = 15
        self._init_view()

    def _init_view(self):
        self.setStyleSheet("background: white;" 
                           "border-radius: {radius};" 
                           "color: black".format(radius=self.size / 2))
        self.setGeometry(0, 0, self.size, self.size)
        self.setText("x")

    def setup_frame(self, frame):
        self.x = frame[2]
        self.y = frame[3]
        self.move(frame[1] - self.size / 2, frame[0] - self.size / 2)
