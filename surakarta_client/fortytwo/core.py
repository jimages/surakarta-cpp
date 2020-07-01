from surakarta.chess import Chess
from surakarta.game import Game
import threading
import socket
from numba import jit


class Core(object):

    def __init__(self):
        self.ai_camp = -1
        self.is_first = False

        # 连接参数
        self.socket = socket.socket()
        self.host_name = "localhost"
        self.port = 8999
        self.socket.connect((self.host_name, self.port))

    def set_match(self, is_first: int):
        """
        下棋之前，设置开局信息
        :param ai_camp: ai的阵营。
        :param is_first:  ai是否先手。
        :return:
        """
        if is_first:
            msg = "1"
        else:
            msg = "0"
        self.socket.send(msg.encode())

    def playing(self, info: dict, callback):
        """
        下棋
        α-β剪枝搜索 or 数据库搜索
        :param game_info: 游戏信息
        :param callback: 回调
        :return:
        """
        thread = threading.Thread(target=self.__playing, args=(info, callback))
        thread.start()


    def __playing(self, info: dict, callback):
        if 'from_y' in info:
            output = '''{x1} {y1} {x2} {y2}'''.format(x1=str(info["from_y"]),
                                                      y1=str(info["from_x"]),
                                                      x2=str(info["to_y"]),
                                                      y2=str(info["to_x"]))
            self.socket.send(output.encode())

        msg = self.socket.recv(2048).decode()
        chess_list = msg.split(" ")

        # 构造Chess
        for i in range(len(chess_list)):
            chess_list[i] = int(chess_list[i])

        from_chess = self.__find_chess(info, chess_list[1], chess_list[0])
        to_chess = self.__find_chess(info, chess_list[3], chess_list[2])
        callback({"from": from_chess, "to": to_chess})


    def __find_chess(self, info:dict, x: int, y: int) -> Chess:
        """
        找到棋子，这里的坐标与外界传入的坐标刚好相反
        :param x: 纵坐标
        :param y: 横坐标
        :return: 棋子
        """
        return info['board'][x][y]
