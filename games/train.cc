/*
 * 自对弈训练程序
 */
#include <vector>
#include <utility>
#include <iostream>
#include <unordered_map>
#include <errno.h>
#include <cstdlib>
#include <cassert>
#include <torch/torch.h>

#include "surakarta.h"
#include "mcts.h"
#include "policy_value_model.h"

PolicyValueNet network;

std::tuple<torch::Tensor, torch::Tensor, torch::Tensor> self_play(PolicyValueNet network) {
    SurakartaState surakarta;
    MCTS::Node<SurakartaState> root(surakarta);
    while (1) {
        root.get_move_probs(surakarta, network);
    }
}

std::tuple<torch::Tensor, torch::Tensor, torch::Tensor> collect_selfplay_data(size_t n_games = 1) {
    for (size_t i = 0; i < n_games; ++i) {
    }
}

int main()
{
    std::ios::sync_with_stdio(false);

    int batch = 5000;
    for (int i = 0; i < batch; ++i) {
        std::cout << "batch: " << i << std::endl;

        collect_selfplay_data(2);
    }
}
