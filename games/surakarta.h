// Author: jiamges123@gmail.com
// Zachary Wang 2019

#include <algorithm>
#include <random>
#include <iterator>
#include <iostream>
#include <utility>
#include <cassert>
#include <omp.h>
#include <cmath>
#include <iostream>
#include <unordered_map>

using std::vector;
using std::pair;
using std::back_inserter;
using std::back_insert_iterator;
using std::count;
using std::cout;
using std::find;
using std::begin;
using std::end;
using std::endl;
using std::cout;
using std::ostream;
using std::istream;
using std::runtime_error;

#include <mcts.h>

#define BOARD_SIZE 6
#define MV_ASSERT(move) assert(move.is_activated);\
    assert(move.is_activated);\
    assert(move.current.first >= 0 && move.current.second < BOARD_SIZE);\
    assert(move.target.first >= 0 && move.target.second < BOARD_SIZE);

#define PR_ASSERT() assert( player_to_move == 1 || player_to_move == 2)

class SurakartaState
{
public:
    enum ChessType : char{ Red = 1, Black = 2, Null = 0};
    typedef struct {
        bool is_activated; // is the movable
        pair<int, int> current, target;
    } Move;
    int player_to_move;

    static const vector<pair<int, int>> outer_loop;
    static const vector<pair<int, int>> inner_loop;
    // a map to speedup the search.
    static const int_fast16_t outer_loop_map[];
    static const int_fast16_t inner_loop_map[];

    static const uint_fast8_t arc_map[];

    static const Move no_move;

    static const char player_markers[3];
    static const ChessType player_chess[3];

    SurakartaState():
        player_to_move(1)
    {
        for(size_t i=0; i < BOARD_SIZE * BOARD_SIZE; ++i) {
            if (i < 2 * BOARD_SIZE) {
                board[i] = ChessType::Black;
            } else if ( i >= 4 * BOARD_SIZE) {
                board[i] = ChessType::Red;
            } else {
                board[i] = ChessType::Null;
            }
        }
    }
    void do_move(Move move, bool is_human = false)
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
                const auto &current = move.current;
                const auto &target = move.target;

                if (!(current.first >= 0 && current.second >= 0 && target.first >= 0
                            && target.second >= 0 && current.first < 6 && current.second < 6
                            && target.first < 6 && target.second < 6))
                    throw runtime_error("下的位置不合法");

                if ((abs(current.first - target.first) > 1 )
                        || ( abs(current.second - target.second) > 1))
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

                for(auto cur_out: cur_pos_out) {
                    for(auto tar_out: tar_pos_out) {
                        if (can_eat(outer_loop.cbegin(), outer_loop.cend(), cur_out, tar_out))
                        {
                            board[tar_out->second * BOARD_SIZE + tar_out->first] = player_chess[player_to_move];
                            board[cur_out->second * BOARD_SIZE + cur_out->first] = ChessType::Null;
                            goto success;
                        }
                    }
                }
                for(auto cur: cur_pos_inn) {
                    for(auto tar: tar_pos_inn) {
                        if (can_eat(inner_loop.cbegin(), inner_loop.cend(), cur, tar))
                        {
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
    player_to_move = 3 - player_to_move;
    return;
}
    bool has_moves() const {
        return !get_moves().empty();
    }
    template<typename RandomEngine>
    void do_random_move(RandomEngine* engine) {
        assert(get_winner() == ChessType::Null);
        Move m;
        while (true) {
            get_moves();
            std::uniform_int_distribution<int> move_idx(static_cast<int>(0),static_cast<int>(moves.size() - 1));
            decltype(move_idx(*engine)) idx;
            idx = move_idx(*engine);
            if (get_winner() == ChessType::Null) {
                m = moves[idx];
                    do_move(m);
                } else {
                    return;
                }
            }
        }
        // Get the result of the match.
    double get_result(int current_player_to_move) const {
        assert(get_winner() != ChessType::Null);
        auto winner = get_winner();
        if (winner == player_chess[current_player_to_move])
            return 0.0;
        else
            return 1.0;
    }

    // Get all available move.
    std::vector<Move>& get_moves() const
    {
        PR_ASSERT();
        if (has_get_moves) {
            return moves;
        } else {
            // 利用局部性原理，在用的时候清除
            moves.clear();
            for (auto row = 0; row < BOARD_SIZE; ++row)
                for (auto col = 0; col < BOARD_SIZE; ++col) {
                    if ( board[row * BOARD_SIZE + col] == player_chess[player_to_move]) {
                        get_valid_move(col, row, back_inserter(moves));
                    }
                }
            has_get_moves = true;
            return moves;
        }
    }
	void print(ostream& out) const
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
				out << player_markers[board[ BOARD_SIZE * row + col]] << ' ';
			}
			out << player_markers[board[BOARD_SIZE * row + BOARD_SIZE - 1]]<< " |" << endl;
		}
		out << "+";
		for (int col = 0; col < BOARD_SIZE - 1; ++col) {
			out << "--";
		}
		out << "-+" << endl;
		out << player_markers[player_to_move] << " to move " << endl << endl;
	}


private:
    template< class InputIt, class T >
    vector<InputIt> find_all( InputIt first, InputIt end, const T& value ) const {
        vector<InputIt> iterators;
        for(auto i = first; i != end; ++i)
            if (*i == value)
                iterators.push_back(i);
        return iterators;
    }
    size_t find_all(bool is_inner,  int_fast16_t x, int_fast16_t y, decltype(inner_loop)::const_iterator iters[]) const {
        const int_fast16_t *map;
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

    bool can_eat(const decltype(inner_loop)::const_iterator begin, const decltype(inner_loop)::const_iterator end,decltype(inner_loop)::const_iterator curr,
            decltype(inner_loop)::const_iterator tart) const {
        auto former = curr > tart? tart: curr;
        auto latter = curr > tart? curr: tart;
        bool flag_former = true;
        bool flag_latter = true;
        // check from former to later.
        for (auto i = former + 1; i != latter; ++i) {
            if (i == end) i = begin;
            // if the former is begin()
            if (i == latter) break;
            if ((board[ i->second * BOARD_SIZE + i->first] != ChessType::Null) && *i != *curr && *i != *tart) {
                flag_former = false;
                break;
            }
        }
        if (flag_former) return true;
        else {
            // check from latter to former if necessary.
            for (auto i = latter + 1; i != former; ++i) {
                if (i == end) i = begin;
                // if the former is begin()
                if (i == former) break;
                if ((board[ i->second * BOARD_SIZE + i->first] != ChessType::Null) && *i != *curr && *i != *tart) {
                    flag_latter = false;
                    break;
                }
            }
        }
        return flag_former | flag_latter;
    }
    // get the winner if we have, return player[0] otherwise.
    ChessType get_winner() const {
        auto begin_iter = begin(board);
        auto end_iter = end(board);
        for (auto i = 1; i <= 2; ++i) {
            if (find(begin_iter, end_iter, player_chess[i]) == end_iter) {
                return player_chess[3 - i];
            }
        }

        // 如歌有一方没有可行动作，则判断谁的子多即可
        auto tmp_moves = get_moves();
        if (tmp_moves.empty()) {
            int num[3] = {0};
            for (auto i = 1; i <= 2; ++i) {
                num[i] = count(begin_iter, end_iter, player_chess[i]);
            }
            if (num[1] > num[2])
                return player_chess[1];
            else
                return player_chess[2];
        }
        return player_chess[0];
    }
    void get_valid_move(int x, int y, back_insert_iterator<vector<Move>> inserter) const {
        // now we check can we eat something.
        bool flag = 0;
        decltype(inner_loop)::const_iterator iters[2];
        auto n = find_all(true, x,y, iters);
        for (auto i = 0; i < n; ++i) {
            auto move = get_valid_eat_one_direction(inner_loop.cbegin(), inner_loop.cend(), iters[i]);
            if (move.is_activated) {
                inserter = std::move(move);
                flag |= 1;
            }
            move = get_valid_eat_one_direction(inner_loop.crbegin(), inner_loop.crend(), make_reverse_iterator(iters[i]) - 1);
            if (move.is_activated)
            {
                inserter = std::move(move);
                flag |= 1;
            }
        }
        n = find_all(false, x,y, iters);
        for (auto i = 0; i < n; ++i) {
            auto move = get_valid_eat_one_direction(outer_loop.cbegin(), outer_loop.cend(), iters[i]);
            if (move.is_activated)
            {
                inserter = std::move(move);
                flag |= 1;
            }
            move = get_valid_eat_one_direction(outer_loop.crbegin(), outer_loop.crend(), make_reverse_iterator(iters[i]) - 1);
            if (move.is_activated)
            {
                inserter = std::move(move);
                flag |= 1;
            }
        }
        if (flag) return;

        // get all valiable moves.
        for (const auto &direc: directions) {
            if (x + direc.first < 6 && x + direc.first >= 0 &&
                    y + direc.second < 6 && y + direc.second >= 0 &&
                    board[ BOARD_SIZE * (y + direc.second) + x + direc.first] == ChessType::Null) {
                inserter = {1,{x,y},{x + direc.first, y + direc.second}};
            }
        }
    }
    template<typename T>
    Move get_valid_eat_one_direction(T begin, T end, T pos) const {
        uint8_t has_passed_arc = false;
        T next;
        for(auto i = pos + 1; i != pos; ++i) {
            if (i == end)
                i = begin;
            if (i == end - 1) {
                next = begin;
            } else {
                next = i + 1;
            }
            if (i == pos) break;
            if (has_passed_arc && board[BOARD_SIZE * i->second + i->first] == player_chess[ 3 - player_to_move])  {
                return {1, {pos->first, pos->second}, {i->first, i->second}};
            } else if (board[BOARD_SIZE * i->second + i->first] == ChessType::Null) {
                if (!has_passed_arc && arc_map[i->first + i->second * BOARD_SIZE] && arc_map[next->first + next->second * BOARD_SIZE])
                    has_passed_arc = true;
            } else {
                if ( *i == *pos) continue;
                else return {0, {pos->first, pos->second}, {i->first, i->second}};
            }
        }
        return {0, {pos->first, pos->second}, {pos->first, pos->second}};
    }
    ChessType board[BOARD_SIZE * BOARD_SIZE];
    static const vector<pair<int, int>> directions;
    mutable bool has_get_moves = false;
    mutable vector<Move> moves;
};
inline ostream& operator<<(ostream& out, const SurakartaState& state)
{
	state.print(out);
	return out;
}
inline istream& operator>>(istream& in, SurakartaState::Move &move)
{
    int cur_x, cur_y;
    int tar_x, tar_y;
    in >> cur_x >> cur_y >> tar_x >> tar_y;
    move.is_activated = 1;
    move.current.first = cur_x;
    move.current.second = cur_y;
    move.target.first = tar_x;
    move.target.second = tar_y;
    return in;
}
inline ostream& operator<<(ostream &out, const SurakartaState::Move &move)
{
    out << move.current.first << " " << move.current.second <<
        " " << move.target.first << " " << move.target.second;
    return out;
}
inline bool operator!=(const SurakartaState::Move lhs, const SurakartaState::Move rhs)
{
    return !(lhs.is_activated && rhs.is_activated && lhs.current == rhs.current && lhs.target == rhs.target);
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
    if (lhs.target> rhs.target)
        return false;

    // if all are the same.
    return false;
}
