/*
 * 自对弈训练程序
 */
#include "helper.h"
#include "mcts.h"
#include "policy_value_model.h"
#include "surakarta.h"
#include <algorithm>
#include <array>
#include <boost/archive/text_iarchive.hpp>
#include <boost/archive/text_oarchive.hpp>
#include <boost/mpi.hpp>
#include <boost/serialization/array.hpp>
#include <boost/serialization/string.hpp>
#include <boost/serialization/utility.hpp>
#include <cassert>
#include <cerrno>
#include <cstdlib>
#include <deque>
#include <iostream>
#include <iterator>
#include <numeric>
#include <random>
#include <torch/torch.h>
#include <tuple>
#include <unordered_map>
#include <utility>
#include <vector>

#define GAME 50000000
#define GAME_LIMIT 300
#define SAMPLE_SIZE 4096
#define GAME_DATA_LIMIT 1000000
#define EVO_BATCH 1

using MCTS::Node;
using MCTS::run_mcts_distribute;
namespace mpi = boost::mpi;

torch::Tensor get_statistc(Node<SurakartaState>* node)
{
    torch::Tensor prob = torch::zeros({ SURAKARTA_ACTION }, torch::kFloat);
    float total = std::accumulate((node->children).begin(), (node->children).end(), 0.0,
        [](const double l, const std::pair<SurakartaState::Move, Node<SurakartaState>*>& r) { return l + r.second->visits; });
    for (auto i = (node->children).begin(); i != (node->children).end(); ++i) {
        prob[move2index(i->first)] = i->second->visits / total;
    }
    return prob.unsqueeze_(0);
}

std::tuple<torch::Tensor, torch::Tensor, torch::Tensor> sample(const torch::Tensor& board,
    const torch::Tensor& mcts, const torch::Tensor& value)
{
    std::random_device rd;
    std::mt19937 g(rd());

    // generate the sample indexs.
    auto size = board.size(0);
    std::vector<size_t> order(size);
    std::array<size_t, SAMPLE_SIZE> order_sampled;

    std::iota(order.begin(), order.end(), 0);
    std::shuffle(order.begin(), order.end(), g);

    std::copy_n(order.begin(), SAMPLE_SIZE, order_sampled.begin());

    // 获得了取样数据的index序列
    torch::Tensor idx = torch::from_blob(order_sampled.data(), { SAMPLE_SIZE }, torch::TensorOptions().dtype(torch::kLong));

    return std::make_tuple(board.index_select(0, idx), mcts.index_select(0, idx), value.index_select(0, idx));
}

void train_server()
{
    mpi::communicator world;
    unsigned int size = world.size();
    std::cout << "total processes: " << size << std::endl;

    unsigned long batch = 1;
    unsigned long game = 1;
    unsigned long long evo_batch = 0;
    torch::Tensor board = torch::empty({ 0 });
    torch::Tensor mcts = torch::empty({ 0 });
    torch::Tensor value = torch::empty({ 0 });

    std::deque<std::pair<int, torch::Tensor>> deque;

    // load the pt if exists
    if (exists("board.pt"))
        torch::load(board, "board.pt");
    if (exists("mcts.pt"))
        torch::load(mcts, "mcts.pt");
    if (exists("value.pt"))
        torch::load(value, "value.pt");

    PolicyValueNet network;

    if (exists("value_policy.pt"))
        network.load_model("value_policy.pt");
    if (exists("optimizer.pt"))
        torch::load(*(network.optimizer), "optimizer.pt");

    std::string state;
    std::array<std::string, 3> dataset;
    mpi::request reqs[2];
    // tag 1 stand for evaluation. tag 2 stand for train data
    reqs[0] = world.irecv(mpi::any_source, 1, state);
    reqs[1] = world.irecv(mpi::any_source, 2, dataset);
    while (true) {
        auto req_pair = mpi::wait_any(std::begin(reqs), std::end(reqs));
        if (req_pair.first.tag() == 1) {
            deque.emplace_back(req_pair.first.source(), torch_deserialize(state));
            reqs[0] = world.irecv(mpi::any_source, 1, state);
        } else {
            evo_batch = 0;
            // get board, probability, value
            torch::Tensor b, p, v;
            b = torch_deserialize(dataset[0]);
            p = torch_deserialize(dataset[1]);
            v = torch_deserialize(dataset[2]);

            board = torch::cat({ b, board });
            mcts = torch::cat({ p, mcts });
            value = torch::cat({ v, value });
            std::cout << "game: " << game << " dataset: " << board.size(0) << " game length:" << v.size(0) << " from process:" << req_pair.first.source() << std::endl;

            if (board.size(0) >= SAMPLE_SIZE) {
                // 等样本数量超过限制的时候，去掉头部的数据。
                std::cout << std::endl;
                unsigned long size = board.size(0);
                if (size > GAME_DATA_LIMIT) {
                    board = board.narrow(0, size - GAME_DATA_LIMIT - 1, GAME_DATA_LIMIT);
                    mcts = mcts.narrow(0, size - GAME_DATA_LIMIT - 1, GAME_DATA_LIMIT);
                    value = value.narrow(0, size - GAME_DATA_LIMIT - 1, GAME_DATA_LIMIT);
                }
                // 训练网络
                torch::Tensor loss, entropy;
                std::tie(b, p, v) = sample(board, mcts, value);
                std::tie(loss, entropy) = network.train_step(b, p, v);
                std::cout << "batch: " << batch++
                          << " loss: " << loss.item<float>()
                          << " entropy: " << entropy.item<float>()
                          << " train dataset size: " << board.size(0)
                          << std::endl;
            }

            // saving the model
            if (batch % 1000 == 0) {
                std::cout << "Saving checkpoint.........." << std::endl;
                network.save_model("value_policy.pt");
                torch::save(board, "board.pt");
                torch::save(mcts, "mcts.pt");
                torch::save(value, "value.pt");
                torch::save(*(network.optimizer), "optimizer.pt");
            }
            ++game;
            reqs[1] = world.irecv(mpi::any_source, 2, dataset);
        }

        // evoluate the state
        if (deque.size() == EVO_BATCH) {
            std::vector<int> source;
            torch::Tensor states = torch::empty({ 0 });

            while (!deque.empty()) {
                auto d = deque.front();
                source.emplace_back(d.first);
                states = at::cat({ states, d.second });
                deque.pop_front();
            }

            torch::Tensor policy, value;
            std::tie(policy, value) = network.policy_value(states);

            assert(policy.size(0) == static_cast<long long>(source.size()));
            // send back the evolution data;
            std::vector<mpi::request> reqs;
            long ind = 0;
            for (auto i = source.begin(); i != source.end(); ++i) {
                reqs.push_back(world.isend(*i, 1, std::make_pair(torch_serialize(policy[ind]), torch_serialize(value[ind]))));
            }
            mpi::wait_all(reqs.begin(), reqs.end());
            evo_batch += EVO_BATCH;
            if (evo_batch == 0) {
                std::cout << evo_batch;
                std::cout.flush();
            } else {
                std::cout << "\r" << evo_batch;
                std::cout.flush();
            }
        }
    }
}

void worker()
{
    mpi::communicator world;
    while (true) {
        torch::Tensor b = torch::zeros({ 0 });
        torch::Tensor p = torch::zeros({ 0 });

        // OK let's play game!
        size_t count = 0;
        SurakartaState game;
        while (game.get_winner() == 0 && count < GAME_LIMIT) {
            Node<SurakartaState> root(game.player_to_move);
            auto move = run_mcts_distribute(&root, game, world, true);
            b = at::cat({ b, game.tensor() }, 0);
            p = at::cat({ p, get_statistc(&root) }, 0);
            game.do_move(move);
            ++count;
        }
        int winner = game.get_winner();
        // play 1 or 2;
        int size = b.size(0);
        auto v = torch::zeros({ size }, torch::kFloat);
        for (int i = 0; i < size; ++i) {
            v[i] = (i + 1) % 2 == winner ? 1.0f : 0.0f;
        }
        std::array<std::string, 3> dataset;
        dataset[0] = torch_serialize(b);
        dataset[1] = torch_serialize(p);
        dataset[2] = torch_serialize(v);
        world.send(0, 2, dataset);
    }
}

int main(int argc, char* argv[])
{
    mpi::environment env(argc, argv);
    mpi::communicator world;
    auto rank = world.rank();
    std::ios::sync_with_stdio(false);

    if (rank == 0) {
        train_server();
    } else {
        worker();
    }
}
