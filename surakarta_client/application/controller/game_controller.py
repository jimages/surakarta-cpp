from application.view import main_window
from application.view import game_view
from PyQt5.QtWidgets import QGridLayout
from surakarta.game import Game
from fortytwo.core import Core
import copy
import datetime


class GameController:

    def __init__(self):
        self.window = main_window.MainWindow()
        self.selected_tag = 0
        self.move_list = []
        self._player = 1  # player的值同 chess.camp 为-1和1
        self._init_view()
        self.game = Game(self._player)
        self.game.reset_board()
        self._is_ai_mode = True
        self._ai_core = Core()
        self._is_game_begin = False
        self._step_num = 0

    def app_launch(self):
        self.window.show()

    def _init_view(self):
        layout = QGridLayout()
        self.game_view = game_view.GameView()
        layout.addWidget(self.game_view)
        self.window.setLayout(layout)
        self.game_view.click_callback = self._did_click_btn
        self.game_view.target_click_callback = self._did_click_target_btn
        self.game_view.chess_move_callback = self._chess_did_move
        self.game_view.game_begin_callback = self._game_begin
        self.game_view.change_mode_callback = self._change_game_mode
        self.game_view.gen_callback = self._gen_chess_board_info

    def _did_click_btn(self, tag):
        if self._is_game_begin is False:
            return
        tag = int(tag)
        # AI模式下就不让对方可以点击了
        if self._is_ai_mode:
            if self._ai_core.ai_camp == -1 and tag < 13:
                return
            elif self._ai_core.ai_camp == 1 and tag > 12:
                return
        # 判断是否轮到当前玩家下棋
        if self._player == -1 and tag > 12:
            return
        elif self._player == 1 and tag < 13:
            return
        # 1. 获得点击棋子所有可下棋位置
        array = self.game.get_chess_moves(tag)
        if len(array) == 0:
            return
        frames = []
        # 2. 获得可下棋位置的界面坐标
        for info in array:
            frames.append(self._get_chess_frame(info["to"].x, info["to"].y))
        # 3. 展示目标位置
        self.game_view.show_targets(frames)
        self.selected_tag = tag
        self.move_list = copy.deepcopy(array)

    def _did_click_target_btn(self, x, y):
        # 1. 去除目标视图
        self.game_view.remove_all_targets()
        # 2. 移动棋子
        self.game_view.move_chess(self.selected_tag, self._get_chess_frame(x, y))

    def _chess_did_move(self, frame):
        for info in self.move_list:
            if info["to"].x == frame[2] and info["to"].y == frame[3]:
                if info["to"].camp != 0:
                    # 吃子的情况需要移除被吃的棋子
                    self.game_view.remove_chess(info["to"].tag)
                # 数据层面移动棋子
                self.game.do_move(info)
                # 修改当前player
                self._player = -self._player
                self.game_view.add_move_info(info["from"].tag,
                                             (info["from"].x, info["from"].y),
                                             (info["to"].x, info["to"].y))
                break
        # 清理使用过的数据
        self.selected_tag = 0
        self.move_list.clear()
        self._step_num += 1
        win, player = self.game.has_winner()
        if win:
            self.game_view.show_game_end(player)
        else:
            self._ai_go_if_need()

    def _game_begin(self, is_ai_first):
        self._is_game_begin = True
        self._ai_core.set_match(is_ai_first)
        if is_ai_first:
            self._ai_core.is_first = True
            self._ai_core.ai_camp = 1
            self._ai_go_if_need()
        else:
            self._ai_core.ai_camp = -1

    def _change_game_mode(self, mode):
        self._is_ai_mode = mode == 2

    def _gen_chess_board_info(self):
        """生成棋谱"""
        date = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        begin_board = "!BBBBBB\n!BBBBBB\n!000000\n!000000\n!RRRRRR\n!RRRRRR\n"
        place = "北京"
        with open("info.txt", "a") as file:
            file.write("#" + date + "|" + place + "\n")
            file.write(begin_board)
            info_list = self.game.record_info_list
            line_number = ["A", "B", "C", "D", "E", "F"]
            for info in info_list:
                line = "B" if info["camp"] == -1 else "R"
                line += str(info["from_x"]) + line_number[info["from_y"]]
                line += "x" if info["is_attack"] is True else "-"
                line += str(info["to_x"]) + line_number[info["to_y"]]
                line += "\n"
                file.write(line)
            file.close()

    def _ai_go_if_need(self):
        if self._is_ai_go():
            # 如果是AI模式下需要AI下棋了
            board_info = self.game.last_board_info
            if board_info is None:
                board_info = {
                    "board": self.game.chess_board,
                    "red_num": 12,
                    "blue_num": 12,
                }
            board_info.update({"step_num": self._step_num})
            self._ai_core.playing(board_info, self._ai_go)

    def _ai_go(self, info: dict):
        self.game_view.remove_all_targets()
        if info is None:
            return
        self.move_list.append(info)
        self.game_view.move_chess(info["from"].tag, self._get_chess_frame(info["to"].x, info["to"].y))

    def _is_ai_go(self):
        if self._is_ai_mode and self._player == self._ai_core.ai_camp:
            return True
        return False

    @staticmethod
    def _get_chess_frame(x, y):
        interval = game_view.INTERVAL
        begin_x = interval * 2
        begin_y = interval * 2
        return begin_x + x * interval, begin_y + y * interval, x, y
