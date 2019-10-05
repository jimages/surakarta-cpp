#include "surakarta.h"
#include "mcts.h"
#include <arpa/inet.h>
#include <cassert>
#include <cerrno>
#include <cstdlib>
#include <iostream>
#include <netinet/in.h>
#include <sys/socket.h>
#include <sys/types.h>
#include <unordered_map>
#include <utility>
#include <vector>

const char SurakartaState::player_markers[] = { '*', 'R', 'B' };
const SurakartaState::Move SurakartaState::no_move = { 0, { 0, 0 }, { 0, 0 } };
const SurakartaState::ChessType SurakartaState::player_chess[] = { SurakartaState::ChessType::Null, SurakartaState::ChessType::Red, SurakartaState::ChessType::Black };
const std::vector<std::pair<int, int>> SurakartaState::outer_loop = { { 1, 0 }, { 0, 1 }, { 1, 1 },
    { 2, 1 }, { 3, 1 }, { 4, 1 }, { 5, 1 }, { 4, 0 }, { 4, 1 }, { 4, 2 }, { 4, 3 }, { 4, 4 }, { 4, 5 },
    { 5, 4 }, { 4, 4 }, { 3, 4 }, { 2, 4 }, { 1, 4 }, { 0, 4 }, { 1, 5 }, { 1, 4 }, { 1, 3 }, { 1, 2 },
    { 1, 1 } };
const std::vector<std::pair<int, int>> SurakartaState::inner_loop = { { 2, 0 }, { 0, 2 }, { 1, 2 },
    { 2, 2 }, { 3, 2 }, { 4, 2 }, { 5, 2 }, { 3, 0 }, { 3, 1 }, { 3, 2 }, { 3, 3 }, { 3, 4 }, { 3, 5 },
    { 5, 3 }, { 4, 3 }, { 3, 3 }, { 2, 3 }, { 1, 3 }, { 0, 3 }, { 2, 5 }, { 2, 4 }, { 2, 3 }, { 2, 2 },
    { 2, 1 } };

const int_fast16_t SurakartaState::outer_loop_map[] = { -1, 0, -1, -1, 7, -1, 1, 87, 3, 4, 168, 6, -1, 22, -1, -1, 9, -1, -1, 21, -1, -1, 10, -1, 18, 564, 16, 15, 366, 13, -1, 19, -1, -1, 12, -1 };
const int_fast16_t SurakartaState::inner_loop_map[] = { -1, -1, 0, 7, -1, -1, -1, -1, 23, 8, -1, -1, 1, 2, 118, 137, 5, 6, 18, 17, 533, 335, 14, 13, -1, -1, 20, 11, -1, -1, -1, -1, 19, 12, -1, -1 };

const vector<pair<int, int>> SurakartaState::directions = { { 1, 0 }, { -1, 0 }, { -1, 1 }, { 0, 1 }, { 1, 1 }, { 1, -1 },
    { 0, -1 }, { -1, -1 } };

const uint_fast8_t SurakartaState::arc_map[] = { 0, 1, 1, 1, 1, 0, 1, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 1, 0, 1, 1, 1, 1, 0 };
const torch::Tensor SurakartaState::outter_loop_mask = torch::from_blob((int[BOARD_SIZE* BOARD_SIZE]) {
                                                                            0, 1, 0, 0, 1, 0,
                                                                            1, 1, 1, 1, 1, 1,
                                                                            0, 1, 0, 0, 1, 0,
                                                                            0, 1, 0, 0, 1, 0,
                                                                            1, 1, 1, 1, 1, 1,
                                                                            0, 1, 0, 0, 1, 0 },
    { BOARD_SIZE, BOARD_SIZE }, torch::TensorOptions().dtype(torch::kInt));
const torch::Tensor SurakartaState::inner_loop_mask = torch::from_blob((int[BOARD_SIZE* BOARD_SIZE]) {
                                                                           0, 0, 1, 1, 0, 0,
                                                                           0, 0, 1, 1, 0, 0,
                                                                           1, 1, 1, 1, 1, 1,
                                                                           1, 1, 1, 1, 1, 1,
                                                                           0, 0, 1, 1, 0, 0,
                                                                           0, 0, 1, 1, 0, 0 },
    { BOARD_SIZE, BOARD_SIZE }, torch::TensorOptions().dtype(torch::kInt));

void SurakartaState::do_move(Move move, bool is_human)
{
    MV_ASSERT(move);
    PR_ASSERT();

    // check the target position can only be held by null or enimy.
    assert(board[pair2index(move.current)] == player_chess[player_to_move]);
    assert(board[pair2index(move.target)] != player_chess[player_to_move]);

    if (is_human) {
        auto moves = get_moves();
        if (std::find(moves.begin(), moves.end(), move) == moves.end())
            throw std::runtime_error("下棋的位置不合法");
        board[pair2index(move.target)] = player_chess[player_to_move];
        board[pair2index(move.current)] = ChessType::Null;
    } else {
        board[pair2index(move.target)] = player_chess[player_to_move];
        board[pair2index(move.current)] = ChessType::Null;
    }

    has_get_moves = false;
    last_move = move;
    player_to_move = 3 - player_to_move;
    return;
}
void SurakartaState::get_valid_move(int x, int y, back_insert_iterator<vector<Move>> inserter, bool only_eat) const
{
    // now we check can we eat something.
    bool flag = false;
    decltype(inner_loop)::const_iterator iters[2];
    int n = find_all(true, x, y, iters);
    for (auto i = 0; i < n; ++i) {
        auto move = get_valid_eat_one_direction(inner_loop.cbegin(), inner_loop.cend(), iters[i]);
        if (move.is_activated) {
            inserter = move;
            flag |= true;
        }
        move = get_valid_eat_one_direction(inner_loop.crbegin(), inner_loop.crend(), make_reverse_iterator(iters[i]) - 1);
        if (move.is_activated) {
            inserter = move;
            flag |= true;
        }
    }
    n = find_all(false, x, y, iters);
    for (auto i = 0; i < n; ++i) {
        auto move = get_valid_eat_one_direction(outer_loop.cbegin(), outer_loop.cend(), iters[i]);
        if (move.is_activated) {
            inserter = move;
            flag |= true;
        }
        move = get_valid_eat_one_direction(outer_loop.crbegin(), outer_loop.crend(), make_reverse_iterator(iters[i]) - 1);
        if (move.is_activated) {
            inserter = move;
            flag |= true;
        }
    }

    if (only_eat && flag)
        return;

    // get all valiable moves.
    for (const auto& direc : directions) {
        if (x + direc.first < 6 && x + direc.first >= 0 && y + direc.second < 6 && y + direc.second >= 0 && board[BOARD_SIZE * (y + direc.second) + x + direc.first] == ChessType::Null) {
            inserter = { 1, { x, y }, { x + direc.first, y + direc.second } };
        }
    }
}
// Get all available move.
std::vector<SurakartaState::Move>& SurakartaState::get_moves(bool only_eat) const
{
    PR_ASSERT();
    if (has_get_moves && this->only_eat == only_eat) {
        return moves;
    } else {
        // 利用局部性原理，在用的时候清除
        moves.clear();
        for (auto row = 0; row < BOARD_SIZE; ++row)
            for (auto col = 0; col < BOARD_SIZE; ++col) {
                if (board[row * BOARD_SIZE + col] == player_chess[player_to_move]) {
                    get_valid_move(col, row, back_inserter(moves), only_eat);
                }
            }
        has_get_moves = true;
        this->only_eat = only_eat;
        return moves;
    }
}
// get the winner if we have, return player[0] otherwise.
int SurakartaState::get_winner() const
{
    auto begin_iter = begin(board);
    auto end_iter = end(board);
    for (auto i = 1; i <= 2; ++i) {
        if (find(begin_iter, end_iter, player_chess[i]) == end_iter) {
            return 3 - i;
        }
    }

    // 如果双方都有子，则判断谁的子多即可
    size_t num[3] = { 0 };
    for (size_t i = 1; i <= 2; ++i) {
        num[i] = std::count(begin_iter, end_iter, player_chess[i]);
    }
    // 0 表示双方子数相同，给0即可
    if (num[1] == num[2])
        return 0;
    if (num[1] > num[2])
        return 1;
    else
        return 2;
}
bool SurakartaState::terminal() const
{
    auto begin_iter = begin(board);
    auto end_iter = end(board);
    for (auto i = 1; i <= 2; ++i) {
        if (find(begin_iter, end_iter, player_chess[i]) == end_iter) {
            return true;
        }
    }
    return false;
}

// 检查在内外环上有几个对于的吃子的位置
size_t SurakartaState::find_all(bool is_inner, int_fast16_t x, int_fast16_t y, decltype(inner_loop)::const_iterator iters[]) const
{
    const int_fast16_t* map;
    decltype(inner_loop)::const_iterator target_iters;
    if (is_inner) {
        map = inner_loop_map;
        target_iters = inner_loop.cbegin();
    } else {
        map = outer_loop_map;
        target_iters = outer_loop.cbegin();
    }
    int_fast16_t target = map[x + y * BOARD_SIZE];
    if (target == -1)
        return 0;
    else if (target > 23) {
        iters[0] = target_iters + (target & 0x1F);
        iters[1] = target_iters + ((target & 0x3E0) >> 5);
        return 2;
    } else {
        iters[0] = target_iters + (target & 0x1F);
        return 1;
    }
}
torch::Tensor SurakartaState::tensor() const
{
    // 构建一个int类型的棋盘,然后再转换成float，这里主要是为了方便计算
    int boardInt[BOARD_SIZE * BOARD_SIZE];
    for (size_t i = 0; i < BOARD_SIZE * BOARD_SIZE; ++i)
        boardInt[i] = static_cast<int>(board[i]);

    torch::Tensor state = torch::zeros({ 9, BOARD_SIZE, BOARD_SIZE }, torch::TensorOptions().dtype(torch::kInt));
    torch::Tensor boardTensor = torch::from_blob(boardInt, { BOARD_SIZE, BOARD_SIZE }, torch::TensorOptions().dtype(torch::kInt));

    // 双方棋子的位置棋子的位置
    state[0] = (boardTensor == player_to_move).to(torch::TensorOptions().dtype(torch::kInt));
    state[1] = (boardTensor == (3 - player_to_move)).to(torch::TensorOptions().dtype(torch::kInt));
    // 外环的棋子
    state[2] = torch::__and__(state[0], outter_loop_mask);
    state[3] = torch::__and__(state[1], outter_loop_mask);
    // 内环的棋子
    state[4] = torch::__and__(state[0], inner_loop_mask);
    state[5] = torch::__and__(state[1], inner_loop_mask);
    // 对方上一步移动的棋子
    if (last_move != no_move && last_move.is_activated) {
        state[6][last_move.current.second][last_move.current.first] = 1;
        state[7][last_move.target.second][last_move.target.first] = 1;
    } else {
        state[6] = torch::zeros({ BOARD_SIZE, BOARD_SIZE }, torch::TensorOptions().dtype(torch::kInt));
        state[7] = torch::zeros({ BOARD_SIZE, BOARD_SIZE }, torch::TensorOptions().dtype(torch::kInt));
    }
    // 我方是否为先手，先手为1，否则不为先手
    if (player_to_move == 1)
        state[8] = torch::ones({ BOARD_SIZE, BOARD_SIZE }, torch::TensorOptions().dtype(torch::kInt));
    else
        state[8] = torch::zeros({ BOARD_SIZE, BOARD_SIZE }, torch::TensorOptions().dtype(torch::kInt));

    return state.to(torch::kFloat).unsqueeze_(0);
}
