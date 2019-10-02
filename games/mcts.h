#ifndef MCTS_HEADER_PETTER
#define MCTS_HEADER_PETTER
#include <cassert>
//
// Zachary Wang 2019
// jiamges123@gmail.com
//
// AlphaZero[1] for surakarta.
//
// Base on Petter's work[2].
//
// Petter Strandmark 2013
// petter.strandmark@gmail.com
//
// Originally based on Python code at
// http://mcts.ai/code/python.html
//
#include <algorithm>
#include <boost/archive/text_iarchive.hpp>
#include <boost/archive/text_oarchive.hpp>
#include <boost/mpi.hpp>
#include <boost/serialization/utility.hpp>
#include <cmath>
#include <cstdlib>
#include <future>
#include <iomanip>
#include <iostream>
#include <map>
#include <memory>
#include <random>
#include <sstream>
#include <string>
#include <torch/torch.h>
#include <vector>

#include "constant.h"
#include "helper.h"
#include "policy_value_model.h"

#define PB_C_BASE 19652.0f
#define PB_C_INIT 1.25f
#ifdef NDEBUG
#define SIMULATION 800
#else
#define SIMULATION 100
#endif // NDEBUG

// mcts simulation in match mode.
#define SIMULATION_MATCH 2000

//
//
// [1] Silver, D., Hubert, T., Schrittwieser, J., Antonoglou, I., Lai, M., Guez, A., … Hassabis, D. (2018).
//     A general reinforcement learning algorithm that masters chess, shogi, and Go through self-play.
//     Science, 362(6419), 1140–1144. https://doi.org/10.1126/science.aar6404u
//
// [2] Prtter Strandmark 2013
//     https://github.com/PetterS/monte-carlo-tree-search
//

namespace mpi = boost::mpi;
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

    std::map<Move, Node<State>*> children;
    const Move move;
    const int player_to_move;
    Node* const parent;

    int visits = 0;
    float value_sum = 0.0;
    float P = 0.0;

    Node(int to_move, float prior = 0.0)
        : move(State::no_move)
        , player_to_move(to_move)
        , parent(nullptr)
        , P(prior)
    {
    }

    Node(const Node&) = delete;

    ~Node()
    {
        for (auto& child : children) {
            delete child.second;
        }
    }

    std::pair<Move, Node<State>*> best_child() const
    {
        assert(!children.empty());
        return *std::max_element(children.begin(), children.end(),
            [](const std::pair<Move, Node<State>*>& a,
                const std::pair<Move, Node<State>*>& b) { return a.second->ucb_score() < b.second->ucb_score(); });
    }

    std::pair<Move, Node<State>*> best_action() const
    {
        assert(!children.empty());
        return *std::max_element(children.begin(), children.end(),
            [](const std::pair<Move, Node<State>*>& a,
                const std::pair<Move, Node<State>*>& b) { return a.second->visits < b.second->visits; });
    }

    bool expanded() const
    {
        return !(children.empty());
    }

    Node<State>* add_child(int player2move, const Move& move, float prior)
    {
        auto node = new Node(player2move, move, this, prior);
        children[move] = node;
        assert(!children.empty());

        return node;
    }

    float value() const
    {
        if (visits == 0)
            return 0;
        return value_sum / visits;
    }

    float ucb_score() const
    {
        assert(parent != nullptr);

        float pb_c = std::log((static_cast<float>(parent->visits) + PB_C_BASE + 1) / PB_C_BASE) + PB_C_INIT;
        pb_c *= std::sqrt(static_cast<float>(parent->visits) / (static_cast<float>(visits) + 1));

        return pb_c * P + value();
    }

    void add_exploration_noise()
    {
        assert(!children.empty());
        std::random_device rd;
        std::gamma_distribution<float> gamma(0.3);
        for (auto i = children.begin(); i != children.end(); ++i) {
            i->second->P = i->second->P * 0.75 + gamma(rd) * 0.25;
        }
    }

private:
    Node(int player_to_move, const Move& move_, Node* parent_, float prior)
        : move(move_)
        , player_to_move(player_to_move)
        , parent(parent_)
        , visits(0)
        , P(prior)
    {
    }
};

/////////////////////////////////////////////////////////
/////////////////////////////////////////////////////////
inline std::pair<torch::Tensor, torch::Tensor> distribute_policy_value(const torch::Tensor& state, mpi::communicator world)
{
    std::string str;
    std::pair<std::string, std::string> data;
    int rank = world.rank();
    world.send(rank % (EVA_SERVER_NUM) + 1, 3, torch_serialize(state));
    world.recv(rank % (EVA_SERVER_NUM) + 1, 4, data);

    return { torch_deserialize(data.first).unsqueeze(0), torch_deserialize(data.second).unsqueeze(0) };
}

template <typename State>
float evaluate(
    Node<State>* node, const State& state, mpi::communicator world, bool only_eat)
{
    torch::Tensor policy, value;
    std::tie(policy, value) = distribute_policy_value(state.tensor(), world);

    // 确认是否进入了cpu
    assert(policy.device() == torch::kCPU);
    assert(value.device() == torch::kCPU);

    for (auto& move : state.get_moves(only_eat)) {
        // First we get the location from policy.
        node->add_child(3 - state.player_to_move, move, (policy[0][move2index(move)]).template item<float>());
    }

    return value.item<float>();
}
template <typename State>
float evaluate(
    Node<State>* node, const State& state, PolicyValueNet& network, bool only_eat)
{
    torch::Tensor policy, value;
    std::tie(policy, value) = network.policy_value(state.tensor());

    // 确认是否进入了cpu
    assert(policy.device() == torch::kCPU);
    assert(value.device() == torch::kCPU);

    for (auto& move : state.get_moves(only_eat)) {
        // First we get the location from policy.
        node->add_child(3 - state.player_to_move, move, (policy[0][move2index(move)]).template item<float>());
    }

    return value.item<float>();
}

template <typename State>
void backpropagate(
    Node<State>* leaf, int to_play, float value)
{
    while (leaf != nullptr) {
        leaf->value_sum += leaf->player_to_move == to_play ? value : (1.0 - value);
        leaf->visits++;
        leaf = leaf->parent;
    }
}

template <typename State>
typename State::Move run_mcts_distribute(Node<State>* root, const State& state, mpi::communicator world, bool is_selfplay = false, bool only_eat = false)
{
    assert(!root->expanded());
    assert(root != nullptr);

    evaluate(root, state, world, only_eat);
    if (is_selfplay)
        root->add_exploration_noise();

    for (int i = 0; i < SIMULATION; ++i) {
        auto node = root;
        auto game = state;

        while (node->expanded()) {
            typename State::Move move;
            std::tie(move, node) = node->best_child();
            game.do_move(move);
        }
        float value = evaluate(node, game, world, only_eat);
        backpropagate(node, game.player_to_move, value);
    }

    return root->best_action().first;
}
template <typename State>
typename State::Move run_mcts(Node<State>* root, const State& state, PolicyValueNet& netowrk, bool only_eat = false)
{
    assert(!root->expanded());
    assert(root != nullptr);

    evaluate(root, state, netowrk, only_eat);

    for (int i = 0; i < SIMULATION_MATCH; ++i) {
        auto node = root;
        auto game = state;

        while (node->expanded()) {
            typename State::Move move;
            std::tie(move, node) = node->best_child();
            game.do_move(move);
        }
        float value = evaluate(node, game, netowrk, only_eat);
        backpropagate(node, game.player_to_move, value);
    }

    return root->best_action().first;
}

}
#endif
