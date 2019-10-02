#include "mcts.h"
#include "surakarta.h"
#include <arpa/inet.h>
#include <cassert>
#include <cstdlib>
#include <errno.h>
#include <iostream>
#include <netinet/in.h>
#include <sys/socket.h>
#include <sys/types.h>
#include <unordered_map>
#include <utility>
#include <vector>

#define PORT 8999
#define BUFFER 2048

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

int main()
{
    try {
        auto serverfd = socket(AF_INET, SOCK_STREAM, 0);
        if (serverfd == -1) {
            perror(strerror(errno));
            exit(EXIT_FAILURE);
        }

        decltype(serverfd) fd;
        struct sockaddr_in src_addr, des_addr;

        bool counter_first = false;
        bool should_move = false;

        if (!counter_first) {
            should_move = true;
        }

        memset(&src_addr, '\0', sizeof(src_addr));

        src_addr.sin_family = AF_INET;
        src_addr.sin_addr.s_addr = INADDR_ANY;
        src_addr.sin_port = htons(PORT);

        socklen_t addrlen = sizeof(src_addr);
        if (bind(serverfd, (struct sockaddr*)&src_addr, sizeof(src_addr)) < 0) {
            perror(strerror(errno));
            exit(EXIT_FAILURE);
        }
        if (listen(serverfd, 1) != 0) {
            perror(strerror(errno));
            exit(EXIT_FAILURE);
        }
        if ((fd = accept(serverfd, (struct sockaddr*)&des_addr, &addrlen)) < 0) {
            perror(strerror(errno));
            exit(EXIT_FAILURE);
        }
        if (counter_first)
            send(fd, static_cast<const void*>("1"), static_cast<size_t>(1), 0);

        SurakartaState state;
        PolicyValueNet network;
        state.player_to_move = 1;
        while (state.has_moves()) {
            cout << endl
                 << "State: " << state << endl;

            SurakartaState::Move move = SurakartaState::no_move;
            if (should_move) {
                while (true) {
                    try {
                        MCTS::Node<SurakartaState> root(state.player_to_move);
                        move = MCTS::run_mcts(&root, state, network);
                        std::cout << "we move: " << move;
                        state.do_move(move);
                        tosocket(fd, move);
                        break;
                    } catch (std::exception& e) {
                        cout << "Invalid move." << endl;
                        cout << e.what() << std::endl;
                    }
                }
            } else {
                while (true) {
                    move = fromsocket(fd);
                    try {
                        std::cout << "opponent move: " << move;
                        state.do_move(move);
                        break;
                    } catch (std::exception&) {
                        cout << "Invalid move." << endl;
                    }
                }
            }
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
        return 0;
    } catch (std::runtime_error& error) {
        std::cerr << "ERROR: " << error.what() << std::endl;
        return -1;
    }
}
