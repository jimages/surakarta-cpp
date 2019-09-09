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
    assert(board[move.current.second * BOARD_SIZE + move.current.first]
        == player_chess[player_to_move]);
    assert(board[move.target.second * BOARD_SIZE + move.target.first]
        != player_chess[player_to_move]);

    if (is_human) {
        if (board[move.target.second * BOARD_SIZE + move.target.first] == ChessType::Null) {
            // we check the available position
            const auto& current = move.current;
            const auto& target = move.target;

            if (!(current.first >= 0 && current.second >= 0 && target.first >= 0
                    && target.second >= 0 && current.first < 6 && current.second < 6
                    && target.first < 6 && target.second < 6))
                throw runtime_error("下的位置不合法");

            if ((abs(current.first - target.first) > 1)
                || (abs(current.second - target.second) > 1))
                throw runtime_error("下的位置不合法");

            board[target.second * BOARD_SIZE + target.first] = player_chess[player_to_move];
            board[current.second * BOARD_SIZE + current.first] = ChessType::Null;

        } else if (board[move.target.second * BOARD_SIZE + move.target.first]
            == player_chess[3 - player_to_move]) {
            // check can we eat the certain postion

            // check the current position
            auto cur_pos_out = find_all(outer_loop.cbegin(), outer_loop.cend(),
                std::make_pair(move.current.first, move.current.second));
            auto cur_pos_inn = find_all(inner_loop.cbegin(), inner_loop.cend(),
                std::make_pair(move.current.first, move.current.second));

            // check the target position.
            auto tar_pos_out = find_all(outer_loop.cbegin(), outer_loop.cend(),
                std::make_pair(move.target.first, move.target.second));
            auto tar_pos_inn = find_all(inner_loop.cbegin(), inner_loop.cend(),
                std::make_pair(move.target.first, move.target.second));

            for (auto cur_out : cur_pos_out) {
                for (auto tar_out : tar_pos_out) {
                    if (can_eat(outer_loop.cbegin(), outer_loop.cend(), cur_out, tar_out)) {
                        board[tar_out->second * BOARD_SIZE + tar_out->first] = player_chess[player_to_move];
                        board[cur_out->second * BOARD_SIZE + cur_out->first] = ChessType::Null;
                        goto success;
                    }
                }
            }
            for (auto cur : cur_pos_inn) {
                for (auto tar : tar_pos_inn) {
                    if (can_eat(inner_loop.cbegin(), inner_loop.cend(), cur, tar)) {
                        board[tar->second * BOARD_SIZE + tar->first] = player_chess[player_to_move];
                        board[cur->second * BOARD_SIZE + cur->first] = ChessType::Null;
                        goto success;
                    }
                }
            }
            throw runtime_error("下的位置不合法");
        } else {
            throw runtime_error("下的位置不合法");
        }
    } else {
        board[move.target.second * BOARD_SIZE + move.target.first] = player_chess[player_to_move];
        board[move.current.second * BOARD_SIZE + move.current.first] = ChessType::Null;
    }

success:
    has_get_moves = false;
    last_move = move;
    player_to_move = 3 - player_to_move;
    return;
}
void SurakartaState::get_valid_move(int x, int y, back_insert_iterator<vector<Move>> inserter) const
{
    // now we check can we eat something.
    bool flag = 0;
    decltype(inner_loop)::const_iterator iters[2];
    int n = find_all(true, x, y, iters);
    for (auto i = 0; i < n; ++i) {
        auto move = get_valid_eat_one_direction(inner_loop.cbegin(), inner_loop.cend(), iters[i]);
        if (move.is_activated) {
            inserter = std::move(move);
            flag |= 1;
        }
        move = get_valid_eat_one_direction(inner_loop.crbegin(), inner_loop.crend(), make_reverse_iterator(iters[i]) - 1);
        if (move.is_activated) {
            inserter = std::move(move);
            flag |= 1;
        }
    }
    n = find_all(false, x, y, iters);
    for (auto i = 0; i < n; ++i) {
        auto move = get_valid_eat_one_direction(outer_loop.cbegin(), outer_loop.cend(), iters[i]);
        if (move.is_activated) {
            inserter = std::move(move);
            flag |= 1;
        }
        move = get_valid_eat_one_direction(outer_loop.crbegin(), outer_loop.crend(), make_reverse_iterator(iters[i]) - 1);
        if (move.is_activated) {
            inserter = std::move(move);
            flag |= 1;
        }
    }
    if (flag)
        return;

    // get all valiable moves.
    for (const auto& direc : directions) {
        if (x + direc.first < 6 && x + direc.first >= 0 && y + direc.second < 6 && y + direc.second >= 0 && board[BOARD_SIZE * (y + direc.second) + x + direc.first] == ChessType::Null) {
            inserter = { 1, { x, y }, { x + direc.first, y + direc.second } };
        }
    }
}
void SurakartaState::print(ostream& out) const
{
    out << endl;
    // print the first line.
    out << "  ";
    for (int col = 0; col < BOARD_SIZE - 1; ++col) {
        out << col << ' ';
    }
    // the last columns
    out << BOARD_SIZE - 1 << endl;

    // for the second line.
    for (int row = 0; row < BOARD_SIZE; ++row) {
        out << row << " ";
        for (int col = 0; col < BOARD_SIZE - 1; ++col) {
            out << player_markers[board[BOARD_SIZE * row + col]] << ' ';
        }
        out << player_markers[board[BOARD_SIZE * row + BOARD_SIZE - 1]] << " |" << endl;
    }
    out << "+";
    for (int col = 0; col < BOARD_SIZE - 1; ++col) {
        out << "--";
    }
    out << "-+" << endl;
    out << player_markers[player_to_move] << " to move " << endl
        << endl;
}
// Get all available move.
std::vector<SurakartaState::Move>& SurakartaState::get_moves() const
{
    PR_ASSERT();
    if (has_get_moves) {
        return moves;
    } else {
        // 利用局部性原理，在用的时候清除
        moves.clear();
        for (auto row = 0; row < BOARD_SIZE; ++row)
            for (auto col = 0; col < BOARD_SIZE; ++col) {
                if (board[row * BOARD_SIZE + col] == player_chess[player_to_move]) {
                    get_valid_move(col, row, back_inserter(moves));
                }
            }
        has_get_moves = true;
        return moves;
    }
}
bool SurakartaState::can_eat(
    const decltype(inner_loop)::const_iterator begin,
    const decltype(inner_loop)::const_iterator end,
    decltype(inner_loop)::const_iterator curr,
    decltype(inner_loop)::const_iterator tart) const
{
    auto former = curr > tart ? tart : curr;
    auto latter = curr > tart ? curr : tart;
    bool flag_former = true;
    bool flag_latter = true;
    // check from former to later.
    for (auto i = former + 1; i != latter; ++i) {
        if (i == end)
            i = begin;
        // if the former is begin()
        if (i == latter)
            break;
        if ((board[i->second * BOARD_SIZE + i->first] != ChessType::Null) && *i != *curr && *i != *tart) {
            flag_former = false;
            break;
        }
    }
    if (flag_former)
        return true;
    else {
        // check from latter to former if necessary.
        for (auto i = latter + 1; i != former; ++i) {
            if (i == end)
                i = begin;
            // if the former is begin()
            if (i == former)
                break;
            if ((board[i->second * BOARD_SIZE + i->first] != ChessType::Null) && *i != *curr && *i != *tart) {
                flag_latter = false;
                break;
            }
        }
    }
    return flag_former | flag_latter;
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

    // 如果有一方没有可行动作，则判断谁的子多即可
    auto tmp_moves = get_moves();
    if (tmp_moves.empty()) {
        int num[3] = { 0 };
        for (auto i = 1; i <= 2; ++i) {
            num[i] = count(begin_iter, end_iter, player_chess[i]);
        }
        if (num[1] > num[2])
            return 1;
        else
            return 2;
    }
    // 0 means the game is not terminated.
    return 0;
}
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
        state[6][last_move.current.first][last_move.current.second] = 1;
        state[7][last_move.target.first][last_move.target.second] = 1;
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
