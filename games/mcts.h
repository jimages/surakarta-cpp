#ifndef MCTS_HEADER_PETTER
#define MCTS_HEADER_PETTER
#include <cassert>
//
// Zachary Wang 2019
// jiamges123@gmail.com
//
// AlphaZero for surakarta.
//
// Base on Petter's work.
//
// Petter Strandmark 2013
// petter.strandmark@gmail.com
//
// Originally based on Python code at
// http://mcts.ai/code/python.html
//

namespace MCTS {
struct MCTSOptions {
    size_t max_simulation = 800;
    bool verbose = false;
};

//
//
// [1] Chaslot, G. M. B., Winands, M. H., & van Den Herik, H. J. (2008).
//     Parallel monte-carlo tree search. In Computers and Games (pp.
//     60-71). Springer Berlin Heidelberg.
//

#include <algorithm>
#include <cstdlib>
#include <future>
#include <iomanip>
#include <iostream>
#include <map>
#include <memory>
#include <random>
#include <set>
#include <sstream>
#include <string>
#include <thread>
#include <torch/torch.h>
#include <vector>

#include "policy_value_model.h"

#ifdef USE_OPENMP
#include <omp.h>
#endif

namespace MCTS {
    using std::cerr;
    using std::endl;
    using std::size_t;
    using std::vector;

    //
    // This class is used to build the game tree. The root is created by the users and
    // the rest of the tree is created by add_node.
    //
    template <typename State>
    class Node {
    public:
        using Move = typename State::Move;

        std::map<typename State::Move, Node*> children;
        const Move move;
        const int player_to_move;

        int visits = 0;
        float Q = 0.0;
        float U = 0.0;
        float P = 0.0;

        Node(const State& state, float prior = 0.0)
            : move(State::no_move)
            , parent(nullptr)
            , player_to_move(state.player_to_move)
            , P(prior)
        {
        }

        Node(const Node&) = delete;

        ~Node()
        {
            for (auto child : children) {
                delete child;
            }
        }

        Node<State>* best_child() const
        {
            assert(!children.empty());

            return *std::max_element(children.begin(), children.end(),
                [](Node* a, Node* b) { return a->visits < b->visits; });
            ;
        }

        Node<State>* select() const
        {
            assert(!children.empty());
            for (auto child : children) {
                child->UCT_score = double(child->wins) / double(child->visits) + std::sqrt(2.0 * std::log(double(this->visits)) / child->visits);
            }

            return *std::max_element(children.begin(), children.end(),
                [](Node* a, Node* b) { return a->UCT_score < b->UCT_score; });
        }
        Node<State>* add_child(int player2move, const Move& move, float prior)
        {
            auto node = new Node(player2move, move, this, prior);
            children[move] = node;
            assert(!children.empty());

            return node;
        }

        void update(double result)
        {
            visits++;

            //double my_wins = wins.load();
            //while ( ! wins.compare_exchange_strong(my_wins, my_wins + result));
        }

        bool is_leaf() const
        {
            return !children.empty();
        }

        std::string to_string() const
        {
            std::stringstream sout;
            sout << "["
                 << "P" << 3 - player_to_move << " "
                 << "M:" << move << " ";
            return sout.str();
        }

        std::string tree_to_string(int max_depth = 1000000, int indent = 0) const
        {
            if (indent >= max_depth) {
                return "";
            }

            std::string s = indent_string(indent) + to_string();
            for (auto child : children) {
                s += child->tree_to_string(max_depth, indent + 1);
            }
            return s;
        }

    private:
        Node* const parent;

        Node(int player_to_move, const Move& move_, Node* parent_, float prior)
            : move(move_)
            , parent(parent_)
            , player_to_move(player_to_move)
            , visits(0)
            , Q(0)
            , U(0)
            , P(prior)
        {
        }

        std::string indent_string(int indent) const
        {
            std::string s = "";
            for (int i = 1; i <= indent; ++i) {
                s += "| ";
            }
            return s;
        }
    };

    /////////////////////////////////////////////////////////
    /////////////////////////////////////////////////////////

    template <typename State>
    void evaluate(
        Node<State>* node, const State& state, const PolicyValueNet& network)
    {
        assert(node->is_leaf());
        assert(state.has_moves());

        torch::Tensor policy, value;
        std::tie(policy, value) = network.policy_value(state.tensor());

        for (auto& move : state.get_moves()) {
            // First we get the location from policy.
            node->add_child(3 - state.player_to_move, move, policy[move2index(move)]);
        }
    }

    template <typename State>
    void add_exploration_noise(Node<typename State>* root)
    {
        assert(!root.children.empty());
        std::gamma_distribution<float> gamma(0.3);
        for (auto i = root->children.begin(); i != root->children.end(); ++i) {
            (*i)->P = (*i)->P * 0.75 + gamma() * 0.25;
        }
    }
}

#endif
