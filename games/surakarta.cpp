#include "surakarta.h"
#include <arpa/inet.h>
#include <cassert>
#include <cstdlib>
#include <errno.h>
#include <iostream>
#include <iterator>
#include <mcts.h>
#include <netinet/in.h>
#include <sys/socket.h>
#include <sys/types.h>
#include <utility>
#include <vector>

const char SurakartaState::player_markers[] = { '*', 'R', 'B' };
const SurakartaState::ChessType SurakartaState::player_chess[] = { SurakartaState::ChessType::Null, SurakartaState::ChessType::Red, SurakartaState::ChessType::Black };

const SurakartaState::Move SurakartaState::no_move = { 0, { 0, 0 }, { 0, 0 } };
const vector<pair<int, int>> SurakartaState::directions = { { 1, 0 }, { -1, 0 }, { -1, 1 }, { 0, 1 }, { 1, 1 }, { 1, -1 },
    { 0, -1 }, { -1, -1 } };
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
void SurakartaState::get_valid_move(int x, int y, back_insert_iterator<vector<Move>> inserter) const
{
    // now we use the recursion algorithm.
    // in corner
    if (!((x == 0 && y == 0) || (x == 0 && y == 5)
            || (x == 5 && y == 0) || (x == 5 && y == 5))) {
        vector<pair<int, int>> pos_list;
        get_eat_move({ x, y }, std::back_inserter(pos_list), Direction::Left, player_chess[player_to_move], 0);
        get_eat_move({ x, y }, std::back_inserter(pos_list), Direction::Up, player_chess[player_to_move], 0);
        get_eat_move({ x, y }, std::back_inserter(pos_list), Direction::Right, player_chess[player_to_move], 0);
        get_eat_move({ x, y }, std::back_inserter(pos_list), Direction::Down, player_chess[player_to_move], 0);
        for (auto& pos : pos_list) {
            inserter = { 1, { x, y }, { pos.first, pos.second } };
        }
    }
    // get all valiable moves.
    for (const auto& direc : directions) {
        if (x + direc.first < 6 && x + direc.first >= 0 && y + direc.second < 6 && y + direc.second >= 0 && board[BOARD_SIZE * (y + direc.second) + x + direc.first] == ChessType::Null) {
            inserter = { 1, { x, y }, { x + direc.first, y + direc.second } };
        }
    }
}
void SurakartaState::get_eat_move(pair<int, int> pos, back_insert_iterator<vector<pair<int, int>>> inserter, Direction dir, ChessType chess, int arc_count) const
{
    int x = pos.first;
    int y = pos.second;
    // now we use the recursion algorithm.
    // in corner
    if ((x == 0 && y == 0) || (x == 0 && y == 5)
        || (x == 5 && y == 0) || (x == 5 && y == 5))
        return;
    // recursion
    if (!get_offset(x, y, dir)) {
        // passing an arc
        static const pair<int, int> pos_arr[] = {
            { 1, -1 }, { 2, -1 }, { 2, 6 }, { 1, 6 },
            { 4, -1 }, { 3, -1 }, { 3, 6 }, { 4, 6 },
            { -1, 1 }, { -1, 2 }, { 6, 2 }, { 6, 1 },
            { -1, 4 }, { -1, 3 }, { 6, 3 }, { 6, 4 }
        };
        auto x_dir = y <= 2 ? Direction::Down : Direction::Up;
        auto y_dir = x <= 2 ? Direction::Right : Direction::Left;
        if (x == -1) {
            return get_eat_move(pos_arr[y - 1], inserter, x_dir, chess, arc_count + 1);
        } else if (x == 6) {
            return get_eat_move(pos_arr[y + 3], inserter, x_dir, chess, arc_count + 1);
        } else if (y == -1) {
            return get_eat_move(pos_arr[x + 7], inserter, y_dir, chess, arc_count + 1);
        } else { // y == 6
            return get_eat_move(pos_arr[x + 11], inserter, y_dir, chess, arc_count + 1);
        }
    } else {
        if (board[y * BOARD_SIZE + x] == chess) {
            return;
        } else if (board[y * BOARD_SIZE+ x] == ChessType::Null) {
            return get_eat_move({ x, y }, inserter, dir, chess, arc_count);
        } else {
            if (arc_count)
                inserter = std::make_pair(x, y);
        }
    }
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
    player1_options.max_time = 30.0;
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
