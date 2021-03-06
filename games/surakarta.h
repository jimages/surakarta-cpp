// Author: jiamges123@gmail.com
// Zachary Wang 2019
#pragma once

#include <omp.h>
#include <spdlog/sinks/basic_file_sink.h>
#include <spdlog/spdlog.h>
#include <torch/torch.h>

#include <algorithm>
#include <cassert>
#include <cmath>
#include <iostream>
#include <iterator>
#include <random>
#include <unordered_map>
#include <utility>

using std::back_insert_iterator;
using std::back_inserter;
using std::begin;
using std::count;
using std::cout;
using std::end;
using std::endl;
using std::find;
using std::istream;
using std::ostream;
using std::pair;
using std::runtime_error;
using std::vector;

#include "mcts.h"

#define BOARD_SIZE       6
#define SURAKARTA_ACTION 1296

#define MV_ASSERT(move)                                                        \
    assert((move).is_activated);                                               \
    assert((move).is_activated);                                               \
    assert((move).current.first >= 0 && move.current.second < BOARD_SIZE);     \
    assert((move).target.first >= 0 && move.target.second < BOARD_SIZE);
#define PR_ASSERT() assert(player_to_move == 1 || player_to_move == 2)

inline size_t pair2index(const std::pair<int, int>& pair)
{
    return BOARD_SIZE * pair.second + pair.first;
}

class SurakartaState
{
public:
    enum ChessType : char
    {
        Red   = 1,
        Black = 2,
        Null  = 0
    };
    struct surakarta_move
    {
        bool is_activated;  // is the movable
        pair<int, int> current, target;
    };
    typedef surakarta_move Move;

    static const vector<pair<int, int>> outer_loop;
    static const vector<pair<int, int>> inner_loop;
    // a map to speedup the search.
    static const int_fast16_t outer_loop_map[];
    static const int_fast16_t inner_loop_map[];

    static const uint_fast8_t arc_map[];
    static const torch::Tensor outter_loop_mask;
    static const torch::Tensor inner_loop_mask;

    static const Move no_move;

    static const char player_markers[3];
    static const ChessType player_chess[3];

    friend inline ostream& operator<<(ostream& out,
                                      const SurakartaState& state);

    int player_to_move;

    SurakartaState() : player_to_move(1)
    {
        for (size_t i = 0; i < BOARD_SIZE * BOARD_SIZE; ++i)
        {
            if (i < 2 * BOARD_SIZE)
            {
                board[i] = ChessType::Black;
            }
            else if (i >= 4 * BOARD_SIZE)
            {
                board[i] = ChessType::Red;
            }
            else
            {
                board[i] = ChessType::Null;
            }
        }
    }
    SurakartaState(const SurakartaState&) = default;

    void do_move(Move move, bool is_human = false);

    torch::Tensor tensor() const;

    bool has_moves(bool eat_only = false) const
    {
        return !get_moves(eat_only).empty();
    }

    // Get the result of the match.
    double get_result(int current_player_to_move) const
    {
        assert(get_winner() != 0);
        auto winner = get_winner();
        if (winner == player_chess[current_player_to_move])
            return 0.0;
        else
            return 1.0;
    }

    std::vector<Move>& get_moves(bool only_eat = false) const;
    int get_winner() const;
    bool terminal() const;

private:
    ChessType board[BOARD_SIZE * BOARD_SIZE];
    Move last_move = no_move;

    static const vector<pair<int, int>> directions;
    mutable bool has_get_moves = false;
    mutable bool only_eat      = false;
    mutable std::vector<Move> moves;

    size_t find_all(bool is_inner, int_fast16_t x, int_fast16_t y,
                    decltype(inner_loop)::const_iterator iters[]) const;
    bool can_eat(const decltype(inner_loop)::const_iterator begin,
                 const decltype(inner_loop)::const_iterator end,
                 decltype(inner_loop)::const_iterator curr,
                 decltype(inner_loop)::const_iterator tart) const;
    void get_valid_move(int x, int y,
                        back_insert_iterator<vector<Move>> inserter,
                        bool only_eat = false) const;

    template <typename T>
    Move get_valid_eat_one_direction(T begin, T end, const T pos) const
    {
        uint_fast32_t has_passed_arc = false;
        T next;
        T i = pos;
        do
        {
            // 设定next的棋子位置。
            if (i == end)
            {
                i    = begin;
                next = begin + 1;
            }
            else if (i == end - 1)
            {
                next = begin;
            }
            else
            {
                next = i + 1;
            }

            // 当遇到对方棋子的时候，如果转了弯，则吃掉对方。
            if (has_passed_arc &&
                board[pair2index(*i)] == player_chess[3 - player_to_move])
            {
                return {1, {pos->first, pos->second}, {i->first, i->second}};
            }

            // next所指向的位置是否经过了环。
            if (!has_passed_arc && arc_map[pair2index(*i)] &&
                arc_map[pair2index(*next)])
                has_passed_arc = true;

            if (board[pair2index(*i)] != ChessType::Null)
            {
                // 如果途径了有棋子的地方,如果是棋子是自己，则跳过，否则退出。
                if (*i == *pos)
                {
                    ++i;
                    continue;
                }
                else
                    return {
                        0, {pos->first, pos->second}, {i->first, i->second}};
            }
            i = next;

        } while (next != pos);

        return {0, {pos->first, pos->second}, {pos->first, pos->second}};
    }

    template <class InputIt, class T>
    vector<InputIt> find_all(InputIt first, InputIt end, const T& value) const
    {
        vector<InputIt> iterators;
        for (auto i = first; i != end; ++i)
            if (*i == value)
                iterators.push_back(i);
        return iterators;
    }
};

inline ostream& operator<<(ostream& out, const SurakartaState& state)
{
    out << endl;
    // print the first line.
    out << "  ";
    for (int col = 0; col < BOARD_SIZE - 1; ++col)
    {
        out << col << ' ';
    }
    // the last columns
    out << BOARD_SIZE - 1 << endl;

    // for the second line.
    for (int row = 0; row < BOARD_SIZE; ++row)
    {
        out << row << " ";
        for (int col = 0; col < BOARD_SIZE - 1; ++col)
        {
            out << SurakartaState::player_markers[state.board[BOARD_SIZE * row +
                                                              col]]
                << ' ';
        }
        out << SurakartaState::player_markers[state.board[BOARD_SIZE * row +
                                                          BOARD_SIZE - 1]]
            << " |" << endl;
    }
    out << "+";
    for (int col = 0; col < BOARD_SIZE - 1; ++col)
    {
        out << "--";
    }
    out << "-+" << endl;
    out << SurakartaState::player_markers[state.player_to_move] << " to move "
        << endl
        << endl;
    return out;
}

inline istream& operator>>(istream& in, SurakartaState::Move& move)
{
    int cur_x, cur_y;
    int tar_x, tar_y;
    in >> cur_x >> cur_y >> tar_x >> tar_y;
    move.is_activated   = true;
    move.current.first  = cur_x;
    move.current.second = cur_y;
    move.target.first   = tar_x;
    move.target.second  = tar_y;
    return in;
}
inline ostream& operator<<(ostream& out, const SurakartaState::Move& move)
{
    out << move.current.first << " " << move.current.second << " "
        << move.target.first << " " << move.target.second;
    return out;
}
inline bool operator!=(const SurakartaState::Move lhs,
                       const SurakartaState::Move rhs)
{
    return !(lhs.is_activated == rhs.is_activated &&
             lhs.current == rhs.current && lhs.target == rhs.target);
}
inline bool operator==(const SurakartaState::Move lhs,
                       const SurakartaState::Move rhs)
{
    return (lhs.is_activated == rhs.is_activated &&
            lhs.current == rhs.current && lhs.target == rhs.target);
}

inline bool operator<(const SurakartaState::Move lhs,
                      const SurakartaState::Move rhs)
{
    // first we compare the is_activate
    if (lhs.is_activated && !rhs.is_activated)
        return false;
    if (!lhs.is_activated && rhs.is_activated)
        return true;

    // if both are activated
    if (lhs.current < rhs.current)
        return true;
    if (lhs.current > rhs.current)
        return false;

    // if both are activated and current are the same.
    if (lhs.target < rhs.target)
        return true;
    if (lhs.target > rhs.target)
        return false;

    // if all are the same.
    return false;
}
inline size_t move2index(const SurakartaState::Move& move)
{
    MV_ASSERT(move);

    return BOARD_SIZE * BOARD_SIZE * pair2index(move.current) +
           pair2index(move.target);
}
