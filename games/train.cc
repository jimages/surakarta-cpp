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
#include <chrono>
#include <cstdlib>
#include <deque>
#include <iostream>
#include <iterator>
#include <mutex>
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
namespace mt = mpi::threading;
std::deque<std::pair<int, torch::Tensor>> receive;
std::deque<std::tuple<int, torch::Tensor, torch::Tensor>> send;
std::deque<std::tuple<torch::Tensor, torch::Tensor, torch::Tensor>> train_data_buffer;
std::mutex rece_mtx, send_mtx, mpi_mtx, net_mtx, tdb_mtx;

inline torch::Tensor get_statistc(Node<SurakartaState>* node)
{
    torch::Tensor prob = torch::zeros({ SURAKARTA_ACTION }, torch::kFloat);
    float total = std::accumulate((node->children).begin(), (node->children).end(), 0.0,
        [](const double l, const std::pair<SurakartaState::Move, Node<SurakartaState>*>& r) { return l + r.second->visits; });
    for (auto i = (node->children).begin(); i != (node->children).end(); ++i) {
        prob[move2index(i->first)] = i->second->visits / total;
    }
    return prob.unsqueeze_(0);
}

inline std::tuple<torch::Tensor, torch::Tensor, torch::Tensor> sample(const torch::Tensor& board,
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

void receiver(mpi::communicator world)
{
    std::string state;
    mpi::request req;
    boost::optional<mpi::status> status;
    decltype(receive) buffer;
    {
        std::lock_guard<std::mutex> lock(mpi_mtx);
        req = world.irecv(mpi::any_source, 1, state);
    }
    mpi_mtx.lock();
    status = req.test();
    mpi_mtx.unlock();
    while (true) {
        if (status) {
            auto board = torch_deserialize(state);
            buffer.emplace_back(status->source(), board);

            mpi_mtx.lock();
            req = world.irecv(mpi::any_source, 1, state);
            mpi_mtx.unlock();
        }

        if (rece_mtx.try_lock()) {
            std::move(buffer.begin(), buffer.end(), back_inserter(receive));
            buffer.clear();
            rece_mtx.unlock();
        }
    }
}
void sender(mpi::communicator world)
{
    std::vector<mpi::request> sending_list;
    decltype(send) buffer;
    while (true) {
        if (send_mtx.try_lock()) {
            std::move(send.begin(), send.end(), back_inserter(buffer));
            sending_list.clear();
            send_mtx.unlock();
        }

        // send the new data.
        if (!buffer.empty()) {
            auto b = buffer.front();
            auto policy_seri = torch_serialize(std::get<1>(b));
            auto value_seri = torch_serialize(std::get<2>(b));
            buffer.pop_front();

            mpi_mtx.lock();
            sending_list.push_back(
                world.isend(std::get<0>(b), 1, std::make_pair(policy_seri, value_seri)));
            mpi_mtx.unlock();
        }
    }

    // test all the requests
    for (auto req = sending_list.begin(); req != sending_list.end();) {
        auto status = req->test();
        if (status) {
            req = sending_list.erase(req);
        } else {
            ++req;
        }
    }
}
void evolution(PolicyValueNet network)
{
    mpi::communicator world;
    decltype(receive) buffer;
    std::vector<int> source;
    // evoluate the state
    std::chrono::high_resolution_clock::time_point time = std::chrono::high_resolution_clock::now();
    double speed;
    while (true) {
        rece_mtx.lock();
        std::move(receive.begin(), receive.end(), back_inserter(buffer));
        receive.clear();
        rece_mtx.unlock();
        if (buffer.size() >= EVO_BATCH) {
            torch::Tensor states = torch::empty({ 0 });
            size_t bat_size = 0;
            while (!buffer.empty() && bat_size < EVO_BATCH) {
                auto d = buffer.front();
                source.emplace_back(d.first);
                states = at::cat({ states, d.second });
                buffer.pop_front();
                ++bat_size;
            }

            torch::Tensor policy_logit, value;
            net_mtx.lock();
            std::tie(policy_logit, value) = network.policy_value(states);
            net_mtx.unlock();

            assert(policy_logit.size(0) == static_cast<long long>(source.size()));
            long ind = 0;
            send_mtx.lock();
            for (auto i = source.begin(); i != source.end(); ++i) {
                send.emplace_back(*i, policy_logit[ind], value[ind]);
                ++ind;
            }
            speed = EVO_BATCH / (std::chrono::duration_cast<std::chrono::duration<double>>(std::chrono::high_resolution_clock::now() - time)).count();
            time = std::chrono::high_resolution_clock::now();
            std::cout << "\r" << speed << "/s";
            std::cout.flush();
        }
    }
}
void collect_dataset(mpi::communicator world)
{
    boost::optional<mpi::status> status;
    std::array<std::string, 3> dataset;
    mpi::request req = world.irecv(mpi::any_source, 2, dataset);
    while (true) {
        mpi_mtx.lock();
        status = req.test();
        mpi_mtx.unlock();
        if (status) {
            // get board, probability, value
            torch::Tensor b, p, v;
            b = torch_deserialize(dataset[0]);
            p = torch_deserialize(dataset[1]);
            v = torch_deserialize(dataset[2]);
            tdb_mtx.lock();
            train_data_buffer.emplace_back(b, p, v);
            tdb_mtx.unlock();
            mpi_mtx.lock();
            req = world.irecv(mpi::any_source, 2, dataset);
            mpi_mtx.unlock();
        }
    }
}
void train(PolicyValueNet network, mpi::communicator world)
{
    torch::Tensor board = torch::empty({ 0 });
    torch::Tensor mcts = torch::empty({ 0 });
    torch::Tensor value = torch::empty({ 0 });
    torch::Tensor b, p, v;
    if (exists("board.pt"))
        torch::load(board, "board.pt");
    if (exists("mcts.pt"))
        torch::load(mcts, "mcts.pt");
    if (exists("value.pt"))
        torch::load(value, "value.pt");

    unsigned long batch = 1;
    unsigned long game = 1;
    while (true) {

        // 将缓冲区中的数据填入到数据集中
        tdb_mtx.lock();
        bool is_ept = train_data_buffer.empty();
        tdb_mtx.unlock();
        while (!is_ept) {
            tdb_mtx.lock();
            auto bpv = train_data_buffer.front();
            board = torch::cat({ std::get<0>(bpv), board });
            mcts = torch::cat({ std::get<1>(bpv), mcts });
            value = torch::cat({ std::get<2>(bpv), value });
            train_data_buffer.pop_front();
            tdb_mtx.unlock();
            std::cout << std::endl;
            std::cout << "game: " << game
                      << " dataset: " << board.size(0)
                      << " game length:" << std::get<0>(bpv).size(0)
                      << std::endl;

            if (board.size(0) >= SAMPLE_SIZE) {
                // 等样本数量超过限制的时候，去掉头部的数据。
                unsigned long size = board.size(0);
                if (size > GAME_DATA_LIMIT) {
                    board = board.narrow(0, size - GAME_DATA_LIMIT - 1, GAME_DATA_LIMIT);
                    mcts = mcts.narrow(0, size - GAME_DATA_LIMIT - 1, GAME_DATA_LIMIT);
                    value = value.narrow(0, size - GAME_DATA_LIMIT - 1, GAME_DATA_LIMIT);
                }
                // 训练网络
                torch::Tensor loss, entropy;
                std::tie(b, p, v) = sample(board, mcts, value);
                net_mtx.lock();
                std::tie(loss, entropy) = network.train_step(b, p, v);
                net_mtx.unlock();
                std::cout << "batch: " << batch++
                          << " loss: " << loss.item<float>()
                          << " entropy: " << entropy.item<float>()
                          << " train dataset size: " << board.size(0)
                          << std::endl;
            }

            // saving the model
            if (batch % 100 == 0) {
                std::cout << "Saving checkpoint.........." << std::endl;
                net_mtx.lock();
                network.save_model("value_policy.pt");
                torch::save(board, "board.pt");
                torch::save(mcts, "mcts.pt");
                torch::save(value, "value.pt");
                torch::save(*(network.optimizer), "optimizer.pt");
                net_mtx.unlock();
            }
            ++game;
        }
    }
}

void server()
{
    mpi::communicator world;
    PolicyValueNet network;
    net_mtx.lock();
    if (exists("value_policy.pt"))
        network.load_model("value_policy.pt");
    if (exists("optimizer.pt"))
        torch::load(*(network.optimizer), "optimizer.pt");
    net_mtx.unlock();
    std::thread(evolution, network);
    std::thread(train, network, world);
    receiver(world);
    sender(world);
    collect_dataset(world);
}

void worker()
{
    mpi::communicator world;
    std::vector<mpi::request> d_trans_queue;
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
            v[i] = (i + 1) % 2 == winner ? 1.0F : 0.0F;
        }
        std::array<std::string, 3> dataset;
        dataset[0] = torch_serialize(b);
        dataset[1] = torch_serialize(p);
        dataset[2] = torch_serialize(v);
        d_trans_queue.push_back(world.isend(0, 2, dataset));
        for (auto req = d_trans_queue.begin(); req != d_trans_queue.end();) {
            auto status = req->test();
            if (status) {
                req = d_trans_queue.erase(req);
            } else {
                ++req;
            }
        }
    }
}

int main(int argc, char* argv[])
{
    mpi::environment env(argc, argv, mt::funneled);
    if (env.thread_level() < mt::funneled) {
        std::cerr << "Fuck!!!" << std::endl;
        env.abort(-1);
        std::abort();
    }
    mpi::communicator world;
    int rank = world.rank();
    std::ios::sync_with_stdio(false);

    if (rank == 0) {
        server();
    } else {
        worker();
    }
    return 0;
}
