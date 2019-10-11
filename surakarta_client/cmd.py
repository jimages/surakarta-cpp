import logging
import socket

from nemesis.core import Core
from surakarta.chess import Chess
from surakarta.game import Game

AI_CAMP = -1
host_name = "192.168.199.156"
port = 8999


class Cmd(object):

    def __init__(self, ai_camp: int):
        self._game = Game(ai_camp, is_debug=True)
        self._game.reset_board()
        self._ai_core = Core()
        self._ai_core.ai_camp = ai_camp
        self.step_num = 0
        self.socket = socket.socket()
        self.socket.connect((host_name, port))

    def start(self):
        is_ai_first = False

        while True:
            msg = self.socket.recv(2048).decode()

            if len(msg) == 1:
                is_ai_first = True
                self._ai_core.is_first = True
                self._ai_core.ai_camp = 1
                self._game.set_camp(1)

            if is_ai_first:
                is_ai_first = False
                self._ai_go()
                continue

            chess_list = msg.split(" ")

            if len(chess_list) != 4:
                print(msg)
                print("⚠️ 输入错误: 缺少输入参数")
                break

            for i in range(len(chess_list)):
                chess_list[i] = int(chess_list[i])
            from_chess = self.find_chess(chess_list[1], chess_list[0])
            to_chess = self.find_chess(chess_list[3], chess_list[2])

            if from_chess.tag == 0:
                logging.error(msg)
                print("⚠️ 输入错误: 输入位置错误")
                break
            info = {"from": from_chess, "to": to_chess}
            self._game.do_move(info)
            self.step_num += 1
            self._ai_go()
        self.socket.close()

    def find_chess(self, x: int, y: int) -> Chess:
        """
        找到棋子，这里的坐标与外界传入的坐标刚好相反
        :param x: 纵坐标
        :param y: 横坐标
        :return: 棋子
        """
        return self._game.chess_board[x][y]

    def _get_board_info(self) -> dict:
        board_info = self._game.last_board_info
        if board_info is None:
            board_info = {
                "board": self._game.chess_board,
                "red_num": 12,
                "blue_num": 12
            }
        board_info.update({"step_num": self.step_num})
        return board_info

    def _ai_go(self):
        self._ai_core.playing(self._get_board_info(), self._ai_move_callback)

    def _ai_move_callback(self, info: dict):
        self._game.do_move(info)
        self.step_num += 1
        output = '''{x1} {y1} {x2} {y2}'''.format(x1=str(info["from"].y),
                                                  y1=str(info["from"].x),
                                                  x2=str(info["to"].y),
                                                  y2=str(info["to"].x))
        self.socket.send(output.encode())
        print(output)


if __name__ == '__main__':
    cmd = Cmd(AI_CAMP)
    cmd.start()
