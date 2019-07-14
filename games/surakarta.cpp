#include "surakarta.h"
#include <mcts.h>
#include <vector>
#include <utility>
#include <iostream>
#include <unordered_map>
#include <sys/socket.h>
#include <sys/types.h>
#include <netinet/in.h>
#include <arpa/inet.h>
#include <errno.h>
#include <cstdlib>
#include <cassert>
#include <mpi.h>

const char SurakartaState::player_markers[] = {'*', 'R', 'B'};
const SurakartaState::ChessType SurakartaState::player_chess[] = {SurakartaState::ChessType::Null, SurakartaState::ChessType::Red, SurakartaState::ChessType::Black};
const std::vector<std::pair<int, int>> SurakartaState::outer_loop = {{1,0}, {0,1}, {1,1},
    {2,1}, {3,1}, {4,1}, {5,1}, {4,0}, {4,1}, {4,2}, {4,3}, {4,4}, {4,5},
    {5,4}, {4,4}, {3,4}, {2,4}, {1,4}, {0,4}, {1,5}, {1,4}, {1,3}, {1,2},
    {1,1}};
const std::vector<std::pair<int, int>> SurakartaState::inner_loop = {{2, 0}, {0, 2}, {1, 2},
    {2, 2}, {3, 2}, {4, 2}, {5, 2}, {3, 0}, {3, 1}, {3, 2}, {3, 3}, {3, 4}, {3, 5},
    {5, 3}, {4, 3}, {3, 3}, {2, 3}, {1, 3}, {0, 3}, {2, 5}, {2, 4}, {2, 3}, {2, 2},
    {2, 1}};

const SurakartaState::Move SurakartaState::no_move = {0,{0,0},{0,0}};
const int_fast16_t SurakartaState::outer_loop_map[] = {-1, 0, -1, -1, 7, -1, 1, 87, 3, 4, 168, 6, -1, 22, -1, -1, 9, -1, -1, 21, -1, -1, 10, -1, 18, 564, 16, 15, 366, 13, -1, 19, -1, -1, 12, -1};
const int_fast16_t SurakartaState::inner_loop_map[] = {-1, -1, 0, 7, -1, -1, -1, -1, 23, 8, -1, -1, 1, 2, 118, 137, 5, 6, 18, 17, 533, 335, 14, 13, -1, -1, 20, 11, -1, -1, -1, -1, 19, 12, -1, -1};

const vector<pair<int, int>> SurakartaState::directions = {{1, 0}, {-1, 0}, {-1,1}, {0, 1}, {1, 1}, {1, -1},
    {0, -1}, {-1, -1}};

const uint_fast8_t SurakartaState::arc_map[] = {0, 1, 1, 1, 1, 0, 1, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 1, 0, 1, 1, 1, 1, 0};

#define PORT  8999
#define BUFFER 2048

SurakartaState::Move fromsocket(int fd) {
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

void tosocket(int fd, SurakartaState::Move move) {
    std::stringstream m;
    m << move;
    auto str = m.str();
    if (send(fd, str.c_str(), str.length(), 0) == -1) {
        perror(strerror(errno));
        exit(EXIT_FAILURE);
    }
}

int main(int argc, char *argv[])
{
    MPI_Init(&argc, &argv);
	try {
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
        bool net_competition = true;


        if (self_play || !counter_first) {
            should_move = true;
        }

        memset(&src_addr, '\0', sizeof(src_addr));

        src_addr.sin_family= AF_INET;
        src_addr.sin_addr.s_addr = INADDR_ANY;
        src_addr.sin_port = htons( PORT );

        socklen_t addrlen = sizeof(src_addr);
        if (net_competition) {
            if (bind(serverfd, (struct sockaddr *)&src_addr, sizeof(src_addr)) < 0)
                perror(strerror(errno));
            if (listen(serverfd, 1) != 0)
                perror(strerror(errno));
            if ((fd = accept(serverfd, (struct sockaddr *)&des_addr, &addrlen)) < 0)
                perror(strerror(errno));
            if (counter_first)
                send(fd, static_cast<const void *>("1"), static_cast<size_t>(1), 0);
        }

        MCTS::ComputeOptions player1_options, player2_options;
        player1_options.max_time =  10.0;
        player1_options.number_of_threads = 8;
        player1_options.verbose = true;

        player2_options.max_time= 30.0;
        player2_options.number_of_threads = 4;
        player2_options.verbose = true;

        SurakartaState state;
        state.player_to_move = 1;
        while (state.has_moves()) {
            cout << endl << "State: " << state << endl;

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
                            }
                            catch (std::exception& ) {
                                cout << "Invalid move." << endl;
                            }
                    }
            }
            if (!self_play)
                should_move = !should_move;
        }

        std::cout << endl << "Final state: " << state << endl;

        if (state.get_result(2) == 1.0) {
            cout << "Player 1 wins!" << endl;
        }
        else if (state.get_result(1) == 1.0) {
            cout << "Player 2 wins!" << endl;
        }
        else {
            cout << "Nobody wins!" << endl;
        }
        return 0;
	}
	catch (std::runtime_error& error) {
		std::cerr << "ERROR: " << error.what() << std::endl;
		return -1;
	}
    MPI_Finalize();
}
