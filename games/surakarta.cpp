#include "surakarta.h"
#include <arpa/inet.h>
#include <cassert>
#include <cstdlib>
#include <errno.h>
#include <iostream>
#include <mcts.h>
#include <netinet/in.h>
#include <sys/socket.h>
#include <sys/types.h>
#include <utility>
#include <vector>

const char SurakartaState::player_markers[] = { '*', 'R', 'B' };
const SurakartaState::ChessType SurakartaState::player_chess[] = { SurakartaState::ChessType::Null, SurakartaState::ChessType::Red, SurakartaState::ChessType::Black };
const std::vector<std::pair<int, int>> SurakartaState::outer_loop = { { 1, 0 }, { 0, 1 }, { 1, 1 },
    { 2, 1 }, { 3, 1 }, { 4, 1 }, { 5, 1 }, { 4, 0 }, { 4, 1 }, { 4, 2 }, { 4, 3 }, { 4, 4 }, { 4, 5 },
    { 5, 4 }, { 4, 4 }, { 3, 4 }, { 2, 4 }, { 1, 4 }, { 0, 4 }, { 1, 5 }, { 1, 4 }, { 1, 3 }, { 1, 2 },
    { 1, 1 } };
const std::vector<std::pair<int, int>> SurakartaState::inner_loop = { { 2, 0 }, { 0, 2 }, { 1, 2 },
    { 2, 2 }, { 3, 2 }, { 4, 2 }, { 5, 2 }, { 3, 0 }, { 3, 1 }, { 3, 2 }, { 3, 3 }, { 3, 4 }, { 3, 5 },
    { 5, 3 }, { 4, 3 }, { 3, 3 }, { 2, 3 }, { 1, 3 }, { 0, 3 }, { 2, 5 }, { 2, 4 }, { 2, 3 }, { 2, 2 },
    { 2, 1 } };

const SurakartaState::Move SurakartaState::no_move = { 0, { 0, 0 }, { 0, 0 } };
const int_fast16_t SurakartaState::outer_loop_map[] = { -1, 0, -1, -1, 7, -1, 1, 87, 3, 4, 168, 6, -1, 22, -1, -1, 9, -1, -1, 21, -1, -1, 10, -1, 18, 564, 16, 15, 366, 13, -1, 19, -1, -1, 12, -1 };
const int_fast16_t SurakartaState::inner_loop_map[] = { -1, -1, 0, 7, -1, -1, -1, -1, 23, 8, -1, -1, 1, 2, 118, 137, 5, 6, 18, 17, 533, 335, 14, 13, -1, -1, 20, 11, -1, -1, -1, -1, 19, 12, -1, -1 };

const vector<pair<int, int>> SurakartaState::directions = { { 1, 0 }, { -1, 0 }, { -1, 1 }, { 0, 1 }, { 1, 1 }, { 1, -1 },
    { 0, -1 }, { -1, -1 } };

const uint_fast8_t SurakartaState::arc_map[] = { 0, 1, 1, 1, 1, 0, 1, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 1, 0, 1, 1, 1, 1, 0 };

#define PORT 8999
#define BUFFER 2048
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

SurakartaState::Move fromsocket(int fd)
{
    char buffer[BUFFER];
    auto l = recv(fd, buffer, BUFFER, 0);
    if (l == -1) {
        perror(strerror(errno));
        exit(-1);
    }
    assert(l == 7);
    buffer[7] = '\0';
    std::stringstream str;
    str << buffer;
    SurakartaState::Move m = SurakartaState::no_move;
    str >> m;
    return m;
}

void tosocket(int fd, SurakartaState::Move move)
{
    std::stringstream m;
    m << move;
    auto str = m.str();
    if (send(fd, str.c_str(), str.length(), 0) == -1) {
        perror(strerror(errno));
        exit(EXIT_FAILURE);
    }
}

void main_program()
{
    auto serverfd = socket(AF_INET, SOCK_STREAM, 0);
    if (serverfd == -1) {
        perror(strerror(errno));
        exit(EXIT_FAILURE);
    }

    decltype(serverfd) fd;
    struct sockaddr_in src_addr, des_addr;

    bool self_play = false;
    bool counter_first = false;
    bool should_move = false;
    bool net_competition = false;

    if (self_play || !counter_first) {
        should_move = true;
    }

    memset(&src_addr, '\0', sizeof(src_addr));

    src_addr.sin_family = AF_INET;
    src_addr.sin_addr.s_addr = INADDR_ANY;
    src_addr.sin_port = htons(PORT);

    socklen_t addrlen = sizeof(src_addr);
    if (net_competition) {
        if (bind(serverfd, (struct sockaddr*)&src_addr, sizeof(src_addr)) < 0)
            perror(strerror(errno));
        if (listen(serverfd, 1) != 0)
            perror(strerror(errno));
        if ((fd = accept(serverfd, (struct sockaddr*)&des_addr, &addrlen)) < 0)
            perror(strerror(errno));
        if (counter_first)
            send(fd, static_cast<const void*>("1"), static_cast<size_t>(1), 0);
    }

    MCTS::ComputeOptions player1_options, player2_options;
    player1_options.max_time = 10.0;
    player1_options.number_of_threads = 1;
    player1_options.verbose = true;

    player2_options.max_time = 30.0;
    player2_options.number_of_threads = 4;
    player2_options.verbose = true;

    SurakartaState state;
    state.player_to_move = 1;
    while (state.has_moves()) {
        cout << endl
             << "State: " << state << endl;

        SurakartaState::Move move = SurakartaState::no_move;
        if (should_move) {
            move = MCTS::compute_move(state, player1_options);
            state.do_move(move);
            if (net_competition)
                tosocket(fd, move);
        } else {
            while (true) {
                if (net_competition) {
                    move = fromsocket(fd);
                } else {
                    cout << "Input your move: ";
                    std::cin >> move;
                }
                try {
                    std::cout << "counter move: " << move;
                    state.do_move(move);
                    break;
                } catch (std::exception&) {
                    cout << "Invalid move." << endl;
                }
            }
        }
        if (!self_play)
            should_move = !should_move;
    }

    std::cout << endl
              << "Final state: " << state << endl;

    if (state.get_result(2) == 1.0) {
        cout << "Player 1 wins!" << endl;
    } else if (state.get_result(1) == 1.0) {
        cout << "Player 2 wins!" << endl;
    } else {
        cout << "Nobody wins!" << endl;
    }
    return;
}

int main()
{
    try {
        main_program();
    } catch (std::runtime_error& error) {
        std::cerr << "ERROR: " << error.what() << std::endl;
        return 1;
    }
}
