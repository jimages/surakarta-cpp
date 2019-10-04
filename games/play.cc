#include "helper.h"
#include "mcts.h"
#include "surakarta.h"
#include <arpa/inet.h>
#include <cassert>
#include <cstdlib>
#include <errno.h>
#include <iostream>
#include <memory>
#include <netinet/in.h>
#include <sys/socket.h>
#include <sys/types.h>
#include <unordered_map>
#include <utility>
#include <vector>

using MCTS::Node;
using std::make_shared;
using std::shared_ptr;
int main()
{
    try {
        bool counter_first = false;
        bool should_move = false;

        if (!counter_first) {
            should_move = true;
        }

        SurakartaState state;
        PolicyValueNet network;
        if (exists("value_policy.pt"))
            network.load_model("value_policy.pt");
        state.player_to_move = 1;
        int steps = 0;
        shared_ptr<Node<SurakartaState>> root = make_shared<Node<SurakartaState>>(state.player_to_move);
        while (state.has_moves()) {
            cout << endl
                 << "State: " << state << endl;
            SurakartaState::Move move = SurakartaState::no_move;
            if (should_move) {
                while (true) {
                    try {
                        move = MCTS::run_mcts(root, state, network, steps / 2);
                        std::cout << "alphazero move: " << move;
                        state.do_move(move);
                        break;
                    } catch (std::exception& e) {
                        cout << "Invalid move." << endl;
                        cout << e.what() << std::endl;
                    }
                }
            } else {
                while (true) {
                    try {
                        cout << "Input your move: ";
                        std::cin >> move;
                        std::cout << "we move: " << move;
                        state.do_move(move);
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
