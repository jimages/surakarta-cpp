/*
 * 自对弈训练程序
 */
#include <algorithm>
#include <array>
#include <cassert>
#include <cerrno>
#include <cstdlib>
#include <iostream>
#include <numeric>
#include <random>
#include <torch/torch.h>
#include <tuple>
#include <unordered_map>
#include <utility>
#include <vector>

#include "helper.h"
#include "mcts.h"
#include "policy_value_model.h"
#include "surakarta.h"

#define GAME 5000
#define GAME_LIMIT 200
#define SAMPLE_SIZE 100

using MCTS::Node;
using MCTS::run_mcts;

torch::Tensor get_statistc(Node<SurakartaState>* node)
{
    torch::Tensor prob = torch::zeros({ SURAKARTA_ACTION }, torch::kFloat);
    float total = std::accumulate((node->children).begin(), (node->children).end(), 0.0,
        [](const double l, const std::pair<SurakartaState::Move, Node<SurakartaState>*>& r) { return l + r.second->visits; });
    for (auto i = (node->children).begin(); i != (node->children).end(); ++i) {
        prob[move2index(i->first)] = i->second->visits / total;
    }
    return prob;
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
    torch::Tensor idx = torch::from_blob(order_sampled.data(), { SAMPLE_SIZE });

    return std::make_tuple(board.index_select(0, idx), mcts.index_select(0, idx), value.index_select(0, idx));
}

int main()
{
    torch::Tensor board, mcts, value;
    // load the pt if exists
    if (exists("board.pt"))
        torch::load(board, "board.pt");
    if (exists("mcts.pt"))
        torch::load(mcts, "mcts.pt");
    if (exists("value.pt"))
        torch::load(value, "value.pt");
    PolicyValueNet network;
    std::ios::sync_with_stdio(false);

    if (exists("value_policy.pt"))
        network.load_model("value_policy.pt");
    if (exists("optimizer.pt"))
        torch::load(*(network.optimizer), "optimizer.pt");

    int game = 5000;
    int batch = 0;
    for (int i = 0; i < game; ++i) {
        torch::Tensor b, p;
        std::cout << "game: " << i << std::endl;

        // OK let's play game!
        size_t count = 0;
        SurakartaState game;
        while (game.get_winner() == 0 && count < GAME_LIMIT) {
            Node<SurakartaState> root(game.player_to_move);
            auto move = run_mcts(&root, game, network, true);
            b = at::cat({ b, game.tensor().unsqueeze(0) });
            p = at::cat({ p, get_statistc(&root).unsqueeze(0) });
            game.do_move(move);
        }
        int winner = game.get_winner();
        // play 1 or 2;
        int size = b.size(0);
        auto v = torch::zeros({ size }, torch::kFloat);
        for (int i = 0; i < size; ++i) {
            v[i] = (i + 1) % 2 == winner ? 1.0f : 0.0f;
        }

        // in the end cat the value.
        board = torch::cat({ board, b });
        mcts = torch::cat({ mcts, p });
        value = torch::cat({ value, v });

        if (board.size(0) > 100) {
            torch::Tensor b, p, v;
            std::tie(b, p, v) = sample(board, mcts, value);

            // 训练网络
            torch::Tensor loss, entropy;
            std::tie(loss, entropy) = network.train_step(b, p, v);
            std::cout << "batch: " << batch++
                      << "loss: " << loss.item<float>()
                      << "entropy: " << entropy.item<float>()
                      << std::endl;
        }

        if (i % 1000 == 0) {
            network.save_model("value_policy.pt");
            torch::save(board, "board.pt");
            torch::save(mcts, "mcts.pt");
            torch::save(value, "value.pt");
            torch::save(*(network.optimizer), "optimizer.pt");
        }
    }
}
