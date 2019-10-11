import copy
from surakarta.chess import Chess
from numba import jit


class PlayManager(object):

    def __init__(self):
        self._board = None
        self._fly_x = 0
        self._fly_y = 0
        self._fly_path = []

    def get_moves(self, camp: int, board: [[Chess]]) -> [dict]:
        """
        获取所有棋子的所有下棋位置
        :param camp: 阵营
        :param board: 棋盘，由chess对象组成
        :return: 字典数组 {'from': chess, 'to': chess}，value为chess对象，from是谁下棋，to是下棋位置
        """
        if camp == 0:
            raise RuntimeError("Camp must be - 1 or 1!!!")

        self._board = copy.deepcopy(board)
        self._fly_x = 0
        self._fly_y = 0
        self._fly_path = []
        fly_moves = self._create_fly_moves(camp)
        walk_moves = self._create_walk_moves(camp)
        walk_moves.extend(fly_moves)
        return walk_moves

    def get_game_moves(self, chess: Chess, board: [[Chess]]) -> [dict]:
        """
        获取一个棋子的所有下棋位置
        :param chess: 要下棋的棋子
        :param board: 棋盘，由chess对象组成
        :return: 字典数组 {'from': chess, 'to': chess}，value为chess对象，from是谁下棋，to是下棋位置
        """
        self._board = copy.deepcopy(board)
        move_list = []
        walk_list = self._walk_engine(chess.x, chess.y)
        for w in walk_list:
            move_list.append({"from": chess, "to": w})
        fly_list = self._begin_fly(chess.x, chess.y, chess.camp)
        for fly in fly_list:
            move_list.append({"from": chess, "to": fly[-1]})
        return move_list

    @jit
    def _create_walk_moves(self, camp: int) -> [dict]:
        move_list = []
        for i in range(0, 6):
            for j in range(0, 6):
                p = self._board[i][j]
                if p.camp == camp:
                    walk_list = self._walk_engine(p.x, p.y)
                    for w in walk_list:
                        d = {"from": p, "to": w}
                        move_list.append(d)
        return move_list

    @jit
    def _create_fly_moves(self, camp: int) -> [dict]:
        move_list = []
        for i in range(0, 6):
            for j in range(0, 6):
                p = self._board[i][j]
                if p.camp == camp:
                    fly_list = self._begin_fly(p.x, p.y, p.camp)
                    for fly in fly_list:
                        move_list.append({"from": p, "to": fly[-1]})
        return move_list

    def _begin_fly(self, x: int, y: int, camp: int) -> [Chess]:
        finish_fly_path = []
        for i in range(0, 4):
            self._fly_x = x
            self._fly_y = y
            self._fly_engine(x, y, i, camp, False)
            if self._fly_path is not None and len(self._fly_path) > 0:
                finish_fly_path.append(copy.deepcopy(self._fly_path))
                self._fly_path = []
        return finish_fly_path

    # 向四个方向走
    def _can_fly(self, orientation: int):
        # 向上
        if orientation == 0:
            self._fly_x -= 1
            if self._fly_x < 0:
                self._fly_x += 1
                return False
            else:
                return True
        # 向右
        if orientation == 1:
            self._fly_y += 1
            if self._fly_y > 5:
                self._fly_y -= 1
                return False
            else:
                return True
        # 向下
        if orientation == 2:
            self._fly_x += 1
            if self._fly_x > 5:
                self._fly_x -= 1
                return False
            else:
                return True
        # 向左
        if orientation == 3:
            self._fly_y -= 1
            if self._fly_y < 0:
                self._fly_y += 1
                return False
            else:
                return True
        return False

    def _fly_engine(self, x: int, y: int, orientation: int, camp: int, already_fly: bool, depth: int = 0):
        # 在四个角里就不继续了
        if (self._fly_x == 0 and self._fly_y == 0) or (self._fly_x == 5 and self._fly_y == 0) or \
                (self._fly_x == 0 and self._fly_y == 5) or (self._fly_x == 5 and self._fly_y == 5):
            return

        # 此循环为向上下左右四个方向搜索（走）
        while self._can_fly(orientation):
            p = self._board[self._fly_x][self._fly_y]
            if p.camp != 0:
                if p.camp + camp == 0:
                    if already_fly:
                        # already_fly为True时表示已经至少绕过一圈了，可以飞了
                        self._fly_path.append(p)
                    else:
                        # 反之这样就被对方棋子挡住了，这条路不用搜了
                        self._fly_path = []
                    return
                else:
                    if self._fly_x == x and self._fly_y == y:
                        # 最多绕4个小圈就不能再绕了，要不然就停不下来了
                        if depth < 4:
                            continue
                        else:
                            self._fly_path = []
                            return
                    else:
                        self._fly_path = []
                        return

        # 走到四个角里也就不继续了
        if (self._fly_x == 0 and self._fly_y == 0) or (self._fly_x == 5 and self._fly_y == 0) or \
                (self._fly_x == 0 and self._fly_y == 5) or (self._fly_x == 5 and self._fly_y == 5):
            return

        # node是在进圈的那个点
        node = self._board[self._fly_x][self._fly_y]
        # 入圈点进入飞行数组
        self._fly_path.append(node)

        # 获取绕完圈出口的点
        next_node_x, next_node_y = self._pathway_table(self._fly_x, self._fly_y)
        # 获取到了就赋值给self.fly_x和self.fly_y
        if next_node_x != -1 and next_node_y != -1:
            self._fly_x = next_node_x
            self._fly_y = next_node_y

        # next_node是出圈的那个点
        next_node = self._board[self._fly_x][self._fly_y]
        if next_node.camp != 0:
            if next_node.camp + camp == 0:
                # 出圈点刚好可以吃，就可以飞了
                self._fly_path.append(next_node)
            else:
                # 出圈点是自己阵营的棋子，那这条路不用搜了
                self._fly_path = []
            return
        else:
            # 获取下一个出圈点的方向
            orientation = self._direction_table(self._fly_x, self._fly_y)
            # 递归继续搜下一个方向，这个时候肯定碰到对方棋子就肯定能吃了
            if depth < 4:
                self._fly_engine(x, y, orientation, camp, True, depth=depth + 1)
            else:
                # 超过4次递归就说明没有搜到
                self._fly_path = []

    # 8个方向找
    def _walk_engine(self, x, y):
        array = []
        if (x - 1 >= 0) & (y - 1 >= 0) & (x - 1 < 6) & (y - 1 < 6):
            if self._board[x - 1][y - 1].camp == 0:
                array.append(self._board[x - 1][y - 1])
        if (x - 1 >= 0) & (y >= 0) & (x - 1 < 6) & (y < 6):
            if self._board[x - 1][y].camp == 0:
                array.append(self._board[x - 1][y])
        if (x - 1 >= 0) & (y + 1 >= 0) & (x - 1 < 6) & (y + 1 < 6):
            if self._board[x - 1][y + 1].camp == 0:
                array.append(self._board[x - 1][y + 1])
        if (x >= 0) & (y - 1 >= 0) & (x < 6) & (y - 1 < 6):
            if self._board[x][y - 1].camp == 0:
                array.append(self._board[x][y - 1])
        if (x >= 0) & (y + 1 >= 0) & (x < 6) & (y + 1 < 6):
            if self._board[x][y + 1].camp == 0:
                array.append(self._board[x][y + 1])
        if (x + 1 >= 0) & (y - 1 >= 0) & (x + 1 < 6) & (y - 1 < 6):
            if self._board[x + 1][y - 1].camp == 0:
                array.append(self._board[x + 1][y - 1])
        if (x + 1 >= 0) & (y >= 0) & (x + 1 < 6) & (y < 6):
            if self._board[x + 1][y].camp == 0:
                array.append(self._board[x + 1][y])
        if (x + 1 >= 0) & (y + 1 >= 0) & (x + 1 < 6) & (y + 1 < 6):
            if self._board[x + 1][y + 1].camp == 0:
                array.append(self._board[x + 1][y + 1])
        return array

    @classmethod
    def _pathway_table(cls, x, y):
        if x == 0:
            if y == 1:
                return 1, 0
            if y == 2:
                return 2, 0
            if y == 3:
                return 2, 5
            if y == 4:
                return 1, 5
        if x == 1:
            if y == 0:
                return 0, 1
            if y == 5:
                return 0, 4
        if x == 2:
            if y == 0:
                return 0, 2
            if y == 5:
                return 0, 3
        if x == 3:
            if y == 0:
                return 5, 2
            if y == 5:
                return 5, 3
        if x == 4:
            if y == 0:
                return 5, 1
            if y == 5:
                return 5, 4
        if x == 5:
            if y == 1:
                return 4, 0
            if y == 2:
                return 3, 0
            if y == 3:
                return 3, 5
            if y == 4:
                return 4, 5
        return -1, -1

    @classmethod
    def _direction_table(cls, x, y):
        if x == 0:
            if (y == 1) or (y == 2) or (y == 3) or (y == 4):
                return 2
        if x == 1:
            if y == 0:
                return 1
            if y == 5:
                return 3
        if x == 2:
            if y == 0:
                return 1
            if y == 5:
                return 3
        if x == 3:
            if y == 0:
                return 1
            if y == 5:
                return 3
        if x == 4:
            if y == 0:
                return 1
            if y == 5:
                return 3
        if x == 5:
            if (y == 1) or (y == 2) or (y == 3) or (y == 4):
                return 0
        return -1