from PyQt5.QtCore import *
from PyQt5.QtWidgets import *
from PyQt5.QtGui import *
import sip
from application.view.target_button import TargetButton
from application.view.chess_button import ChessButton

INTERVAL = 50
LONG_RADIUS = INTERVAL * 4
SHORT_RADIUS = INTERVAL * 2
CHESS_SIZE = 30


class GameView(QWidget):

    def __init__(self, *__args):
        super().__init__(*__args)
        self.click_callback = None
        self.target_click_callback = None
        self.chess_move_callback = None
        self.game_begin_callback = None
        self.change_mode_callback = None
        self.gen_callback = None
        self.targets = []
        self.chess_list = []
        self._player = -1
        self._is_ai_first_go = False
        self._init_view()
        self._init_timer()

    def _init_view(self):
        self._setup_buttons()
        self.human_radio = QRadioButton("人人", self)
        self.human_radio.setGeometry(INTERVAL * 10, 0, 100, 25)
        self.ai_radio = QRadioButton("人机", self)
        self.ai_radio.setGeometry(INTERVAL * 10 + 100, 0, 100, 25)
        mode_button_group = QButtonGroup(self)
        mode_button_group.addButton(self.human_radio, 1)
        mode_button_group.addButton(self.ai_radio, 2)
        mode_button_group.buttonClicked.connect(self._select_mode_radio)
        self.first_human_radio = QRadioButton("人先手", self)
        self.first_human_radio.setGeometry(INTERVAL * 10, 35, 100, 25)
        self.first_human_radio.hide()
        self.first_ai_radio = QRadioButton("机先手", self)
        self.first_ai_radio.setGeometry(INTERVAL * 10 + 100, 35, 100, 25)
        self.first_ai_radio.hide()
        first_button_group = QButtonGroup(self)
        first_button_group.addButton(self.first_human_radio, 1)
        first_button_group.addButton(self.first_ai_radio, 2)
        first_button_group.buttonClicked.connect(self._select_first_radio)
        self.begin_button = QPushButton(self)
        self.begin_button.setStyleSheet("QPushButton{border-radius: 10; background-color: white; color: black;}"
                                        "QPushButton:hover{background-color: lightgray}")
        self.begin_button.setText("开始")
        self.begin_button.setGeometry(INTERVAL * 10, 70, 200, 25)
        self.begin_button.clicked.connect(self._click_begin_button)
        self.gen_button = QPushButton(self)
        self.gen_button.setStyleSheet("QPushButton{border-radius: 10; background-color: white; color: black;}"
                                      "QPushButton:hover{background-color: lightgray}")
        self.gen_button.setText("生成棋谱")
        self.gen_button.setGeometry(INTERVAL * 10, 100, 200, 25)
        self.gen_button.clicked.connect(self._click_gen_button)
        self.red_time_label = QLabel(self)
        self.red_time_label.setText("00:00")
        self.red_time_label.setStyleSheet("color: red")
        self.red_time_label.setGeometry(INTERVAL * 10, 130, 100, 25)
        self.blue_time_label = QLabel(self)
        self.blue_time_label.setText("00:00")
        self.blue_time_label.setStyleSheet("color: blue")
        self.blue_time_label.setGeometry(INTERVAL * 10 + 100, 130, 100, 25)
        self.list_widget = QListWidget(self)
        self.list_widget.setGeometry(INTERVAL * 10, 160, 200, 300)

    def _init_timer(self):
        self._red_time = 0
        self._blue_time = 0
        self._timer = QTimer(self)
        self._timer.setInterval(1000)
        self._timer.timeout.connect(self._timer_operate)

    def show_game_end(self, player):
        if player == -1:
            message = "红方获胜"
        else:
            message = "蓝方获胜"
        print(message)

    def show_targets(self, frames):
        self.remove_all_targets()
        for frame in frames:
            btn = TargetButton(self)
            btn.setup_frame(frame)
            btn.clicked.connect(self._click_target_btn)
            btn.show()
            self.targets.append(btn)

    def remove_all_targets(self):
        for btn in self.targets:
            btn.hide()
            sip.delete(btn)
        self.targets.clear()

    def remove_chess(self, tag):
        for btn in self.chess_list:
            if btn.tag == tag:
                self.chess_list.remove(btn)
                btn.hide()
                sip.delete(btn)
                break

    def move_chess(self, chess_tag, to_frame):
        self._player = -self._player
        for chess in self.chess_list:
            if chess_tag == chess.tag:
                chess.move(to_frame[1] - CHESS_SIZE / 2, to_frame[0] - CHESS_SIZE / 2)
                # 移动完棋子要回调修改棋盘数据
                self.chess_move_callback(to_frame)
                return

    def add_move_info(self, tag: int, f: tuple, t: tuple):
        text = "tag {tag}: ({fx}, {fy}) -> ({tx}, {ty})".format(tag=tag,
                                                                fx=f[0],
                                                                fy=f[1],
                                                                tx=t[0],
                                                                ty=t[1])
        item = QListWidgetItem(text)
        self.list_widget.addItem(item)

    @pyqtSlot()
    def _click_gen_button(self):
        self.gen_callback()

    @pyqtSlot()
    def _click_btn(self):
        self.click_callback(self.sender().tag)

    @pyqtSlot()
    def _click_target_btn(self):
        self.target_click_callback(self.sender().x, self.sender().y)

    @pyqtSlot()
    def _select_mode_radio(self):
        if self.sender().checkedId() == 1:
            self.first_human_radio.hide()
            self.first_ai_radio.hide()
            self.change_mode_callback(1)
        else:
            self.first_human_radio.show()
            self.first_ai_radio.show()
            self.change_mode_callback(2)

    @pyqtSlot()
    def _click_begin_button(self):
        self._player = 1
        self._timer.start()
        self.begin_button.setEnabled(False)
        self.ai_radio.setEnabled(False)
        self.human_radio.setEnabled(False)
        self.first_human_radio.setEnabled(False)
        self.first_ai_radio.setEnabled(False)
        self.game_begin_callback(self._is_ai_first_go)

    @pyqtSlot()
    def _select_first_radio(self):
        if self.sender().checkedId() == 1:
            self._is_ai_first_go = False
        else:
            self._is_ai_first_go = True

    @pyqtSlot()
    def _timer_operate(self):
        if self._player == -1:
            self._red_time += 1
        else:
            self._blue_time += 1
        time = self._red_time if self._player == -1 else self._blue_time
        m = int(time / 60)
        if m < 10:
            str_m = "0{m}".format(m=m)
        else:
            str_m = str(m)
        s = time - m * 60
        if s < 10:
            str_s = "0{s}".format(s=s)
        else:
            str_s = str(s)
        if self._player == -1:
            self.red_time_label.setText(str_m + ":" + str_s)
        else:
            self.blue_time_label.setText(str_m + ":" + str_s)

    def _setup_buttons(self):
        begin_x = INTERVAL * 2
        begin_y = INTERVAL * 2
        for i in range(0, 24):
            btn = ChessButton(self)
            if i < 6:
                btn.setup_view(True)
                btn.setGeometry(begin_x + INTERVAL * i - CHESS_SIZE / 2,
                                begin_y - CHESS_SIZE / 2,
                                CHESS_SIZE,
                                CHESS_SIZE)
            elif i < 12:
                btn.setup_view(True)
                btn.setGeometry(begin_x + INTERVAL * (i - 6) - CHESS_SIZE / 2,
                                begin_y + INTERVAL - CHESS_SIZE / 2,
                                CHESS_SIZE,
                                CHESS_SIZE)
            elif i < 18:
                btn.setup_view(False)
                btn.setGeometry(begin_x + INTERVAL * (i - 12) - CHESS_SIZE / 2,
                                begin_y + INTERVAL * 4 - CHESS_SIZE / 2,
                                CHESS_SIZE,
                                CHESS_SIZE)
            else:
                btn.setup_view(False)
                btn.setGeometry(begin_x + INTERVAL * (i - 18) - CHESS_SIZE / 2,
                                begin_y + INTERVAL * 5 - CHESS_SIZE / 2,
                                CHESS_SIZE,
                                CHESS_SIZE)
            btn.setText(str(i + 1))
            btn.tag = i + 1
            btn.clicked.connect(self._click_btn)
            self.chess_list.append(btn)

    def paintEvent(self, QPaintEvent):
        painter = QPainter(self)
        painter.setPen(QColor(166, 66, 250))
        # 左上
        painter.drawArc(0, 0, LONG_RADIUS, LONG_RADIUS, 0, 270 * 16)
        painter.drawArc(INTERVAL, INTERVAL, SHORT_RADIUS, SHORT_RADIUS, 0, 270 * 16)

        # 左下
        painter.drawArc(0, INTERVAL * 5, LONG_RADIUS, LONG_RADIUS, 90 * 16, 270 * 16)
        painter.drawArc(INTERVAL, INTERVAL * 6, SHORT_RADIUS, SHORT_RADIUS, 90 * 16, 270 * 16)

        # 右上
        painter.drawArc(INTERVAL * 5, 0, LONG_RADIUS, LONG_RADIUS, -90 * 16, 270 * 16)
        painter.drawArc(INTERVAL * 6, INTERVAL, SHORT_RADIUS, SHORT_RADIUS, -90 * 16, 270 * 16)

        # 右下
        painter.drawArc(INTERVAL * 5, INTERVAL * 5, LONG_RADIUS, LONG_RADIUS, -180 * 16, 270 * 16)
        painter.drawArc(INTERVAL * 6, INTERVAL * 6, SHORT_RADIUS, SHORT_RADIUS, -180 * 16, 270 * 16)

        # 竖线
        painter.drawLine(INTERVAL * 2, INTERVAL * 2, INTERVAL * 2, INTERVAL * 7)
        painter.drawLine(INTERVAL * 3, INTERVAL * 2, INTERVAL * 3, INTERVAL * 7)
        painter.drawLine(INTERVAL * 4, INTERVAL * 2, INTERVAL * 4, INTERVAL * 7)
        painter.drawLine(INTERVAL * 5, INTERVAL * 2, INTERVAL * 5, INTERVAL * 7)
        painter.drawLine(INTERVAL * 6, INTERVAL * 2, INTERVAL * 6, INTERVAL * 7)
        painter.drawLine(INTERVAL * 7, INTERVAL * 2, INTERVAL * 7, INTERVAL * 7)

        # 横线
        painter.drawLine(INTERVAL * 2, INTERVAL * 2, INTERVAL * 7, INTERVAL * 2)
        painter.drawLine(INTERVAL * 2, INTERVAL * 3, INTERVAL * 7, INTERVAL * 3)
        painter.drawLine(INTERVAL * 2, INTERVAL * 4, INTERVAL * 7, INTERVAL * 4)
        painter.drawLine(INTERVAL * 2, INTERVAL * 5, INTERVAL * 7, INTERVAL * 5)
        painter.drawLine(INTERVAL * 2, INTERVAL * 6, INTERVAL * 7, INTERVAL * 6)
        painter.drawLine(INTERVAL * 2, INTERVAL * 7, INTERVAL * 7, INTERVAL * 7)
        painter.drawLine(INTERVAL * 2, INTERVAL * 7, INTERVAL * 7, INTERVAL * 7)
