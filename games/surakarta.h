// Author: jiamges123@gmail.com
// Zachary Wang 2019
#pragma once

#include <algorithm>
#include <cassert>
#include <cmath>
#include <iostream>
#include <iterator>
#include <omp.h>
#include <random>
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

#define BOARD_SIZE 6
#define SURAKARTA_ACTION 1296

#define MV_ASSERT(move)                                                    \
    assert((move).is_activated);                                           \
    assert((move).is_activated);                                           \
    assert((move).current.first >= 0 && move.current.second < BOARD_SIZE); \
    assert((move).target.first >= 0 && move.target.second < BOARD_SIZE);
#define PR_ASSERT() assert(player_to_move == 1 || player_to_move == 2)

inline size_t pair2index(const std::pair<int, int>& pair)
{
    return BOARD_SIZE * pair.second + pair.first;
}

class SurakartaState {
public:
    enum ChessType : char { Red = 1,
        Black = 2,
        Null = 0 };
    struct surakarta_move {
        bool is_activated; // is the movable
        pair<int, int> current, target;
    };
    typedef surakarta_move Move;

    static const Move no_move;

    static const char player_markers[3];
    static const ChessType player_chess[3];

    friend inline ostream& operator<<(ostream& out, const SurakartaState& state);

    int player_to_move;

    SurakartaState()
        : player_to_move(1)
    {
        for (size_t i = 0; i < BOARD_SIZE * BOARD_SIZE; ++i) {
            if (i < 2 * BOARD_SIZE) {
                board[i] = ChessType::Black;
            } else if (i >= 4 * BOARD_SIZE) {
                board[i] = ChessType::Red;
            } else {
                board[i] = ChessType::Null;
            }
        }
    }
    SurakartaState(const SurakartaState&) = default;

    void do_move(Move move, bool is_human = false);

    template <typename RandomEngine>
    void do_random_move(RandomEngine* engine)
    {
        assert(has_moves());
        Move m;
        while (true) {
            get_moves();
            std::uniform_int_distribution<int> move_idx(static_cast<int>(0), static_cast<int>(moves.size() - 1));
            decltype(move_idx(*engine)) idx;
            idx = move_idx(*engine);
            if (has_moves()) {
                m = moves[idx];
                do_move(m);
            } else {
                return;
            }
        }
    }

    bool has_moves() const
    {
        return !get_moves().empty();
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

    std::vector<Move>& get_moves() const;
    int get_winner() const;
    bool terminal() const;

private:
    ChessType board[BOARD_SIZE * BOARD_SIZE];
    Move last_move = no_move;

    static const vector<pair<int, int>> directions;
    mutable bool has_get_moves = false;
    mutable std::vector<Move> moves;
    enum class Direction : char {
        Up = 1,
        Down = 2,
        Left = 4,
        Right = 8,
        LeftUp = 5,
        LeftDown = 6,
        RightUp = 9,
        RightDown = 10
    };

    void get_eat_move(pair<int, int> pos, back_insert_iterator<vector<pair<int, int>>> inserter, Direction dir, ChessType chess, int arc_count) const;
    void get_valid_move(int x, int y, back_insert_iterator<vector<Move>> inserter) const;
    bool get_offset(int& x, int& y, Direction dir) const
    {
        const auto up_value = static_cast<char>(Direction::Up);
        const auto down_value = static_cast<char>(Direction::Down);
        const auto left_value = static_cast<char>(Direction::Left);
        const auto right_value = static_cast<char>(Direction::Right);
        auto dir_value = static_cast<char>(dir);
        if ((dir_value & up_value)) {
            y -= 1;
        } else if ((dir_value & down_value)) {
            y += 1;
        }
        if ((dir_value & left_value)) {
            x -= 1;
        } else if ((dir_value & right_value)) {
            x += 1;
        }
        return x >= 0 && x <= 5 && y >= 0 && y <= 5;
    }
};

inline ostream& operator<<(ostream& out, const SurakartaState& state)
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
            out << SurakartaState::player_markers[state.board[BOARD_SIZE * row + col]] << ' ';
        }
        out << SurakartaState::player_markers[state.board[BOARD_SIZE * row + BOARD_SIZE - 1]] << " |" << endl;
    }
    out << "+";
    for (int col = 0; col < BOARD_SIZE - 1; ++col) {
        out << "--";
    }
    out << "-+" << endl;
    out << SurakartaState::player_markers[state.player_to_move] << " to move " << endl
        << endl;
    return out;
}

inline istream& operator>>(istream& in, SurakartaState::Move& move)
{
    int cur_x, cur_y;
    int tar_x, tar_y;
    in >> cur_x >> cur_y >> tar_x >> tar_y;
    move.is_activated = true;
    move.current.first = cur_x;
    move.current.second = cur_y;
    move.target.first = tar_x;
    move.target.second = tar_y;
    return in;
}
inline ostream& operator<<(ostream& out, const SurakartaState::Move& move)
{
    out << move.current.first << " " << move.current.second << " " << move.target.first << " " << move.target.second;
    return out;
}
inline bool operator!=(const SurakartaState::Move lhs, const SurakartaState::Move rhs)
{
    return !(lhs.is_activated == rhs.is_activated && lhs.current == rhs.current && lhs.target == rhs.target);
}
inline bool operator==(const SurakartaState::Move lhs, const SurakartaState::Move rhs)
{
    return (lhs.is_activated == rhs.is_activated && lhs.current == rhs.current && lhs.target == rhs.target);
}

inline bool operator<(const SurakartaState::Move lhs, const SurakartaState::Move rhs)
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

    return BOARD_SIZE * BOARD_SIZE * pair2index(move.current) + pair2index(move.target);
}
