from PyQt5.QtWidgets import QPushButton


class ChessButton(QPushButton):

    def __init__(self, *__args):
        super().__init__(*__args)
        self.tag = 0
        self.size = 30

    def setup_view(self, is_red):
        if is_red:
            color = "red"
        else:
            color = "blue"
        self.setStyleSheet("background-color: {back};" 
                           "border-radius: {radius};" 
                           "color: white".format(back=color, radius=self.size / 2))
