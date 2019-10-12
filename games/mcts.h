#pragma once
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
#include <iterator>
#include <map>
#include <memory>
#include <mutex>
#include <random>
#include <sstream>
#include <string>
#include <thread>
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
#ifdef NDEBUG
#define SIMULATION_MATCH 20000
#else
#define SIMULATION_MATCH 800
#endif

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
using std::async;
using std::cerr;
using std::endl;
using std::future;
using std::shared_ptr;
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

    std::map<Move, shared_ptr<Node<State>>> children;
    using move_node_tuple = typename decltype(children)::value_type;
    const Move move;
    const int player_to_move;
    Node<State>* parent = nullptr;

    std::recursive_mutex mtx;

    int visits = 0;
    float value_sum = 0.0;
    float P = 0.0;
    double Q = 0.0;

    Node(int to_move, float prior = 0.0)
        : move(State::no_move)
        , player_to_move(to_move)
        , P(prior)
    {
    }
    Node(int player_to_move, const Move& move_, Node* parent_, float prior)
        : move(move_)
        , player_to_move(player_to_move)
        , parent(parent_)
        , P(prior)
    {
    }

    Node(const Node&) = delete;

    move_node_tuple best_child()
    {
        std::lock_guard<std::recursive_mutex> g(mtx);
        visits += VIRTUAL_LOSS;
        assert(!children.empty());
        return *std::max_element(children.begin(), children.end(),
            [](const move_node_tuple& a,
                const move_node_tuple& b) { return a.second->ucb_score() < b.second->ucb_score(); });
    }

    move_node_tuple best_action(uint_fast32_t steps, double temp = 1.0) const
    {
        assert(!children.empty());
        std::vector<move_node_tuple> v(children.begin(), children.end());
        if (temp > 1e-2) {

            std::vector<double> w;
            w.reserve(v.size());
            std::transform(v.begin(), v.end(), std::back_inserter(w), [temp](const move_node_tuple& r) { return std::pow(r.second->visits, 1.0 / temp + 1e-10); });

            // get the action
            std::random_device dev;
            std::mt19937 rd(dev());
            std::discrete_distribution<> dd(w.begin(), w.end());
            long ac_idx = dd(rd);
#ifndef NDEBUG
            double total = std::accumulate(children.begin(), children.end(), 0.0, [](double l, move_node_tuple r) { return l + r.second->visits; });
            double total_t = std::accumulate(w.begin(), w.end(), 0.0);
            std::cout << "total: " << total << '\n';
            for (int i = 0; i < v.size(); ++i) {
                std::cout << "move:" << v[i].first << "  visits:" << v[i].second->visits
                          << "  ratio:" << w[i] / total_t << "  p:" << v[i].second->P
                          << "  v:" << v[i].second->value_sum / v[i].second->visits << '\n';
            }

            std::cout << "we chouse move:" << v[ac_idx].first << "  visits:" << v[ac_idx].second->visits
                      << "  ratio:" << w[ac_idx] / total_t
                      << "  p:" << v[ac_idx].second->P << "  v:" << v[ac_idx].second->value_sum / v[ac_idx].second->visits << '\n';
#endif
            return v[ac_idx];
        } else {
            return *std::max_element(children.begin(), children.end(),
                [](const move_node_tuple& a,
                    const move_node_tuple& b) { return a.second->visits < b.second->visits; });
        }
    }

    bool
    expanded() const
    {
        return !(children.empty());
    }

    shared_ptr<Node<State>> add_child(int player2move, const Move& move, float prior)
    {
        auto node = std::make_shared<Node>(player2move, move, this, prior);
        children[move] = node;

        return node;
    }
    shared_ptr<Node<State>> get_child(const Move& move)
    {
        auto p = children.at(move);
        p->parent = nullptr;
        return p;
    }

    float ucb_score() const
    {
        assert(parent != nullptr);

        // float pb_c = std::log((static_cast<float>(parent->visits) + PB_C_BASE + 1) / PB_C_BASE) + PB_C_INIT;
        float pb_c = 5.0;
        pb_c *= std::sqrt(static_cast<float>(parent->visits)) / (static_cast<float>(visits) + 1);

        return pb_c * P + Q;
    }

    void add_exploration_noise()
    {
        assert(!children.empty());
        std::random_device dev;
        std::mt19937 rd(dev());
        std::gamma_distribution<float> gamma(0.3);
        for (auto i = children.begin(); i != children.end(); ++i) {
            i->second->P = i->second->P * 0.75 + gamma(rd) * 0.25;
        }
    }

private:
};

/////////////////////////////////////////////////////////
/////////////////////////////////////////////////////////
inline std::pair<torch::Tensor, torch::Tensor> distribute_policy_value(const torch::Tensor& state, mpi::communicator world, double& diff)
{
    std::string str;
    std::pair<std::string, std::string> data;
    int rank = world.rank();
    auto t = omp_get_wtime();
    world.send(rank % (EVA_SERVER_NUM) + 1, 3, torch_serialize(state));
    world.recv(rank % (EVA_SERVER_NUM) + 1, 4, data);
    auto e = omp_get_wtime();
    diff += (e - t);

    return { torch_deserialize(data.first).unsqueeze(0), torch_deserialize(data.second).unsqueeze(0) };
}

template <typename State>
float evaluate(
    shared_ptr<Node<State>> node, const State& state, mpi::communicator world, bool only_eat, double& diff)
{
    torch::Tensor policy, value;
    std::tie(policy, value) = distribute_policy_value(state.tensor(), world, diff);

    // 确认是否进入了cpu
    assert(policy.device() == torch::kCPU);
    assert(value.device() == torch::kCPU);
    auto& moves = state.get_moves(only_eat);
    float policy_sum = std::accumulate(moves.begin(), moves.end(), 0.0f, [&policy](float l, const typename State::Move& move) { return l + policy[0][move2index(move)].template item<float>(); });

    node->mtx.lock();
    if (node->expanded()) {
        for (auto& move : moves) {
            // First we get the location from policy.
            node->add_child(3 - state.player_to_move, move, (policy[0][move2index(move)]).template item<float>() / policy_sum);
        }
    }
    node->mtx.unlock();

    return value.item<float>();
}
template <typename State>
float evaluate(
    shared_ptr<Node<State>> node, const State& state, PolicyValueNet& network)
{
    torch::Tensor policy, value;
    std::tie(policy, value) = network.policy_value(state.tensor());
    auto& moves = state.get_moves();
    float policy_sum = std::accumulate(moves.begin(), moves.end(), 0.0f, [&policy](float l, const typename State::Move& move) { return l + policy[0][move2index(move)].template item<float>(); });

    for (auto& move : moves) {
        // First we get the location from policy.
        node->add_child(3 - state.player_to_move, move, (policy[0][move2index(move)]).template item<float>() / policy_sum);
    }

    // 确认是否进入了cpu
    assert(policy.device() == torch::kCPU);
    assert(value.device() == torch::kCPU);

    return value.item<float>();
}

template <typename State>
void backpropagate(
    shared_ptr<Node<State>> l, int to_play, float value)
{
    auto leaf = l.get();
    // for leaf node.
    leaf->mtx.lock();
    leaf->value_sum += leaf->player_to_move == to_play ? value : -value;
    leaf->visits += 1;
    leaf->Q = leaf->value_sum / leaf->visits;
    leaf->mtx.unlock();
    leaf = leaf->parent;

    while (leaf != nullptr) {
        leaf->mtx.lock();
        leaf->value_sum += leaf->player_to_move == to_play ? value : -value;
        leaf->visits -= (VIRTUAL_LOSS - 1);
        leaf->Q = leaf->value_sum / leaf->visits;
        leaf->mtx.unlock();
        leaf = leaf->parent;
    }
}

template <typename State>
typename State::Move run_mcts_distribute(shared_ptr<Node<State>> root, const State& state, mpi::communicator world, const int steps, bool eat_only, double& diff)
{
    assert(root != nullptr);
    assert(root->parent == nullptr);

    evaluate(root, state, world, eat_only, diff);
    root->add_exploration_noise();

    for (int i = 0; i < SIMULATION; ++i) {
        auto node = root;
        auto game = state;

        while (node->expanded()) {
            typename State::Move move;
            std::tie(move, node) = node->best_child();
            game.do_move(move);
        }
        float value = evaluate(node, game, world, eat_only, diff);
        backpropagate(node, game.player_to_move, value);
    }

    return root->best_action(steps, 1.0).first;
}
template <typename State>
shared_ptr<Node<State>> mcts_thread(shared_ptr<Node<State>> root, const State& state, PolicyValueNet& network, const int steps)
{
    assert(root != nullptr);
    assert(root->parent == nullptr);

    root->mtx.lock();
    if (root->children.empty())
        evaluate(root, state, network);
    root->mtx.unlock();
    auto time = omp_get_wtime();

    for (int i = 0; (omp_get_wtime() - time) < 10; ++i) {
        auto node = root;
        auto game = state;

        while (node->expanded()) {
            std::lock_guard<std::recursive_mutex> g(node->mtx);
            typename State::Move move;
            std::tie(move, node) = node->best_child();
            game.do_move(move);
        }
        float value = evaluate(node, game, network);
        backpropagate(node, game.player_to_move, value);
    }
    return root;
}
template <typename State>
typename State::Move run_mcts(shared_ptr<Node<State>> root, const State& state, PolicyValueNet& network, const int steps)
{
    vector<future<shared_ptr<Node<State>>>> root_future;
    for (int i = 0; i < 2; i++) {
        auto func = [root, state, &network, steps]() -> shared_ptr<Node<State>> {
            return mcts_thread(root, state, network, steps);
        };
        root_future.push_back(async(std::launch::async, func));
    }
    for (auto& r : root_future) {
        r.get();
    }
    std::cout << root->visits / 10.0 << "/s\n";
    return root->best_action(steps).first;
}
}
