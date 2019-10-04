#include "helper.h"
#include "mcts.h"
#include "surakarta.h"
#include <arpa/inet.h>
#include <cassert>
#include <cstdlib>
#include <errno.h>
#include <getopt.h>
#include <iostream>
#include <memory>
#include <netinet/in.h>
#include <sys/socket.h>
#include <sys/types.h>
#include <unordered_map>
#include <utility>
#include <vector>

#define BUFFER 2048
#define PORT 8999
const char* default_addres = "localhost:8999";
using MCTS::Node;
using std::make_shared;
using std::shared_ptr;
void usage()
{
    std::cout << "The surakarta AI player program. Zachary Wang (jimages123@gmail.com) Under MIT License.\n"
              << "Usage:\n"
              << "     -h --help               Print this message\n"
              << "     -n                      Play with with network opponent\n"
              << "     -m                      Play with human in the console [default]\n"
              << "     -s                      self play\n"
              << "     -f                      The opponent play first\n"
              << std::flush;
}
SurakartaState::Move fromsocket(const int fd)
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

void tosocket(int fd, const SurakartaState::Move& move)
{
    std::stringstream m;
    m << move;
    auto str = m.str();
    if (send(fd, str.c_str(), str.length(), 0) == -1) {
        perror(strerror(errno));
        exit(EXIT_FAILURE);
    }
}

// get action from human.
auto human_action(const SurakartaState::Move& m) -> SurakartaState::Move
{
    SurakartaState::Move move;
    cout << "Input your move: ";
    std::cin >> move;
    std::cout << "we move: " << move;
    return move;
}

// get action from network.
SurakartaState::Move network_action(const int fd, const SurakartaState::Move& m)
{
    SurakartaState::Move move;
    tosocket(fd, m);
    move = fromsocket(fd);
    std::cout << "the network opponent move: " << move;
    return move;
}

int main(int argc, char* argv[])
{

    int ch;
    bool counter_first = false;
    bool in_network = false;
    bool selfplay = false;
    bool human = true;
    static struct option longopts[] = {
        { "help", no_argument, NULL, 'h' },
        { "network", optional_argument, NULL, 'n' },
        { "human", no_argument, NULL, 'm' },
        { "self", no_argument, NULL, 's' },
        { "first", no_argument, NULL, 'f' },
        { NULL, 0, NULL, 0 }
    };

    // Get the options.
    while ((ch = getopt_long(argc, argv, "hnhsf", longopts, NULL)) != -1) {
        switch (ch) {
        case '?':
        case 'h':
            usage();
            std::exit(EXIT_SUCCESS);
            break;
        case 'n':
            std::cout << "play in net play mode\n";
            in_network = true;
            human = false;
            selfplay = false;
            break;
        case 'm':
            std::cout << "play in human play mode\n";
            in_network = false;
            human = true;
            selfplay = false;
            break;
        case 's':
            std::cout << "play in self play mode\n";
            in_network = false;
            human = false;
            selfplay = true;
            break;
        case 'f':
            counter_first = true;
            break;
        default:
            std::cout << "play in human play mode\n";
        }
    }
    try {
        // Init the model and the state.
        SurakartaState state;
        state.player_to_move = 1;
        PolicyValueNet network;
        if (exists("value_policy.pt"))
            network.load_model("value_policy.pt");

        bool should_move = !counter_first;

        // Init the socket.
        auto serverfd = socket(AF_INET, SOCK_STREAM, 0);
        if (serverfd == -1) {
            perror(strerror(errno));
            exit(EXIT_FAILURE);
        }

        decltype(serverfd) fd;
        struct sockaddr_in src_addr, des_addr;
        if (in_network) {
            std::cout << "Listen to the port:" << PORT << '\n';

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

            // send the signal to the opponent.
            if (counter_first)
                send(fd, static_cast<const void*>("1"), static_cast<size_t>(1), 0);
        }

        int steps = 0;
        shared_ptr<Node<SurakartaState>> root = make_shared<Node<SurakartaState>>(state.player_to_move);
        while (state.has_moves()) {
            cout << endl
                 << "State: " << state << endl;
            SurakartaState::Move move = SurakartaState::no_move;
            if (should_move) {
                while (true) {
                    try {
                        SurakartaState::Move m;
                        m = MCTS::run_mcts(root, state, network, steps / 2);
                        std::cout << "alphazero move: " << m;
                        state.do_move(m);
                        move = m;
                        break;
                    } catch (std::exception& e) {
                        cout << "Invalid move." << endl;
                        cout << e.what() << std::endl;
                    }
                }
            } else {
                while (true) {
                    try {
                        SurakartaState::Move m;
                        if (human)
                            m = human_action(move);
                        if (in_network)
                            m = network_action(fd, move);
                        if (selfplay) {
                            m = MCTS::run_mcts(root, state, network, steps / 2);
                            std::cout << "alphazero move: " << m;
                        }

                        state.do_move(m);
                        move = m;
                        break;
                    } catch (std::exception& e) {
                        cout << "Invalid move." << endl;
                        cout << e.what() << std::endl;
                    }
                }
            }
            root = root->get_child(move);
            should_move = !should_move;
            ++steps;
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
