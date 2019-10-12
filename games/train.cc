/*
 * 自对弈训练程序
 */
#include "constant.h"
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
#include <functional>
#include <iostream>
#include <iterator>
#include <memory>
#include <numeric>
#include <omp.h>
#include <pthread.h>
#include <random>
#include <string>
#include <torch/torch.h>
#include <tuple>
#include <utility>
#include <vector>

using MCTS::Node;
using MCTS::run_mcts_distribute;
namespace mpi = boost::mpi;
namespace mt = mpi::threading;

torch::Tensor get_statistc(std::shared_ptr<Node<SurakartaState>> node)
{
    torch::Tensor prob = torch::zeros({ SURAKARTA_ACTION }, torch::kFloat);
    float total = std::accumulate((node->children).begin(), (node->children).end(), 0.0,
        [](const float l, const Node<SurakartaState>::move_node_tuple& r) { return l + r.second->visits; });
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
    mpi::communicator server = world.split(1);
    unsigned int size = world.size();
    std::cout << "total processes: " << size << std::endl;
    std::cout << "the simulation count:" << SIMULATION << std::endl;
    std::cout << "evoluiton batch size:" << EVO_BATCH << std::endl;

    unsigned long game = 1;
    u_long batch = 1;
    torch::Tensor board = torch::empty({ 0 });
    torch::Tensor mcts = torch::empty({ 0 });
    torch::Tensor value = torch::empty({ 0 });
    u_long totoal_evaluation = 0;
    double time = omp_get_wtime();
    double last_sync_model;
    std::string model_buf;
    bool has_updated = false;

    // load the pt if exists
    if (exists("board.pt"))
        torch::load(board, "board.pt");
    if (exists("mcts.pt"))
        torch::load(mcts, "mcts.pt");
    if (exists("value.pt"))
        torch::load(value, "value.pt");

    PolicyValueNet network;
    std::cout << network.model << std::endl;

    if (exists("value_policy.pt"))
        network.load_model("value_policy.pt");
    if (exists("optimizer.pt"))
        torch::load(*(network.optimizer), "optimizer.pt");

    // brocast the model
    std::cout << "Broadcast the network." << std::endl;
    if (torch::cuda::is_available()) {
        network.model->to(torch::kCPU);
    }
    model_buf = network.serialize();
    if (torch::cuda::is_available()) {
        network.model->to(torch::kCUDA);
    }

    mpi::broadcast(server, model_buf, 0);
    server.barrier();
    last_sync_model = omp_get_wtime();

    std::array<std::string, 3> dataset;
    // tag 1 stand for evaluation. tag 2 stand for train data
    auto req = world.irecv(mpi::any_source, 2, dataset);

    while (true) {
        boost::optional<mpi::status> status;

        if (status = req.test()) {
            // get board, probability, value
            torch::Tensor b, p, v;
            b = torch_deserialize(dataset[0]);
            p = torch_deserialize(dataset[1]);
            v = torch_deserialize(dataset[2]);

            board = torch::cat({ b, board });
            mcts = torch::cat({ p, mcts });
            value = torch::cat({ v, value });
            std::cout << std::endl;
            std::cout << "game: " << game
                      << " dataset: " << board.size(0)
                      << " game length:" << v.size(0)
                      << " from process:" << status->source()
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
                std::tie(loss, entropy) = network.train_step(b, p, v);
                std::cout << "batch: " << batch++
                          << " loss: " << loss.item<float>()
                          << " entropy: " << entropy.item<float>()
                          << " train dataset size: " << board.size(0)
                          << std::endl;
                has_updated = true;
            }

            // saving the model
            if (batch % 100 == 0) {
                std::cout << "Saving checkpoint.........." << std::endl;
                network.save_model("value_policy.pt");
                torch::save(board, "board.pt");
                torch::save(mcts, "mcts.pt");
                torch::save(value, "value.pt");
                torch::save(*(network.optimizer), "optimizer.pt");
            }
            ++game;
            req = world.irecv(mpi::any_source, 2, dataset);
        }
        if ((omp_get_wtime() - time) > TIME_LIMIT) {
            u_long total;
            mpi::reduce(server, totoal_evaluation, total, std::plus<u_long>(), 0);
            std::cout << "\r"
                      << static_cast<double>(total) / (omp_get_wtime() - time)
                      << "/s";
            std::cout.flush();
            time = omp_get_wtime();
            totoal_evaluation = 0;
            if ((omp_get_wtime() - last_sync_model) > 60) {
                if (has_updated) {
                    if (torch::cuda::is_available()) {
                        network.model->to(torch::kCPU);
                    }
                    model_buf = network.serialize();
                    if (torch::cuda::is_available()) {
                        network.model->to(torch::kCUDA);
                    }
                    mpi::broadcast(server, model_buf, 0);
                    last_sync_model = omp_get_wtime();
                    has_updated = false;
                    std::cout << std::endl
                              << "Sync the lastest network." << std::endl;
                } else {
                    model_buf.clear();
                    mpi::broadcast(server, model_buf, 0);
                    last_sync_model = omp_get_wtime();
                    std::cout << std::endl
                              << "No update so skip the fetching." << std::endl;
                }
            }
        }
    }
}
void evoluation_server()
{
    mpi::communicator world;
    mpi::communicator server = world.split(1);

    std::string model_buffer;
    std::string state_buffer;

    auto req = world.irecv(mpi::any_source, 3, state_buffer);
    std::cout << "evaluation proecess rank:" << world.rank() << std::endl;
    boost::optional<mpi::status> status;

    std::deque<int> sender_deque;
    std::vector<mpi::request> d_trans_queue;

    // When run on cpu we should initizlize the model on cpu.
    // So the we assign 1 to totoal_device to bypass the divide_by_zero errors.
    int total_device = torch::cuda::device_count();
    if (!torch::cuda::is_available()) {
        total_device = 1;
    }
    PolicyValueNet net((world.rank()) % total_device);

    // Init the model.
    double time = omp_get_wtime();
    double last_sync_model;

    mpi::broadcast(server, model_buffer, 0);
    net.deserialize(model_buffer);
    server.barrier();
    last_sync_model = omp_get_wtime();

    u_long totoal_evaluation = 0;
    torch::Tensor evolution_batch = torch::empty({ 0 });

    while (true) {
        if (status = req.test()) {
            sender_deque.emplace_back(status->source());
            evolution_batch = torch::cat({ evolution_batch, torch_deserialize(state_buffer) });
            req = world.irecv(mpi::any_source, 3, state_buffer);
        }
        // evoluate the state
        if (sender_deque.size() >= EVO_BATCH) {

            torch::Tensor policy_logit, value;
            std::tie(policy_logit, value) = net.policy_value(evolution_batch);

            assert(policy_logit.size(0) == static_cast<long long>(evolution_batch.size(0)));
            assert(sender_deque.size() == evolution_batch.size(0));

            for (long long ind = 0; ind != evolution_batch.size(0); ++ind) {
                d_trans_queue.push_back(world.isend(sender_deque.at(ind), 4, std::make_pair(torch_serialize(policy_logit[ind]), torch_serialize(value[ind]))));
            }
            sender_deque.clear();
            evolution_batch = torch::empty({ 0 });
            totoal_evaluation += EVO_BATCH;
        }
        if ((omp_get_wtime() - time) > TIME_LIMIT) {
            mpi::reduce(server, totoal_evaluation, std::plus<u_long>(), 0);
            totoal_evaluation = 0;
            time = omp_get_wtime();
            if ((omp_get_wtime() - last_sync_model) > 60) {
                mpi::broadcast(server, model_buffer, 0);
                if (!model_buffer.empty()) {
                    net.deserialize(model_buffer);
                }
                last_sync_model = omp_get_wtime();
            }
        }

        // test all the requests
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

void worker()
{
    mpi::communicator world;
    mpi::communicator client = world.split(2);
    std::vector<mpi::request> d_trans_queue;
    while (true) {
        torch::Tensor b = torch::zeros({ 0 });
        torch::Tensor p = torch::zeros({ 0 });

        // OK let's play game!
        size_t count = 0;
        SurakartaState game;
        auto root = std::make_shared<Node<SurakartaState>>(game.player_to_move);
        auto s_time = omp_get_wtime();
        double diff = 0.0;
        while (!game.terminal() && game.has_moves() && count < GAME_LIMIT) {
            auto board = game.tensor();
            unsigned int equal_count = 0;
            auto move = run_mcts_distribute(root, game, world, count / 2);
            // for long situation.
            for (int i = b.size(0) - 2; i >= 0; i -= 2) {
                if (board[0].slice(0, 0, 2).to(torch::kBool).equal(b[i].slice(0, 0, 2).to(torch::kBool))) {
                    equal_count++;
                }
                if (equal_count >= LONG_SITUATION_THRESHOLD) {
                    std::cout << std::endl;
                    std::cout << "find long situation from process:" << world.rank() << std::endl;
                    goto out;
                }
            }
        out:
            b = at::cat({ b, board }, 0);
            p = at::cat({ p, get_statistc(root) }, 0);
            if (equal_count >= LONG_SITUATION_THRESHOLD)
                goto finish;

            // if in long situation. exit.
            game.do_move(move);
            root = root->get_child(move);
            ++count;
        }
    finish:
        // play 1 or 2, 0 for draw
        int winner = game.get_winner();
        int size = b.size(0);
        auto v = torch::zeros({ size }, torch::kFloat);
        if (winner != 0) {
            for (int i = 0; i < size; ++i) {
                v[i] = (i % 2 + 1) == winner ? 1.0F : -1.0F;
            }
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
    mpi::environment env(argc, argv, mt::multiple);
    if (env.thread_level() < mt::multiple) {
        std::cerr << "unsupport the thread environment.\n";
        env.abort(-1);
    }
    mpi::communicator world;
    auto rank = world.rank();
    std::ios::sync_with_stdio(false);

    if (rank == 0) {
        train_server();
    } else if (rank <= EVA_SERVER_NUM) {
        evoluation_server();
    } else {
        worker();
    }
}
