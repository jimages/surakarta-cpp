#ifndef POLICY_VALUE_MODEL_H
#define POLICY_VALUE_MODEL_H
#include <torch/torch.h>
#include <utility>

struct NetImpl : torch::nn::Module {
    static const int_fast32_t width = 6;
    static const int_fast32_t height = 6;
    NetImpl();
    std::pair<torch::Tensor, torch::Tensor> forward(torch::Tensor x);

    /*
     * 输入特征:
     *  我方棋局的形态 6 * 6
     *  对方局面的形态 6 * 6
     *  我方在外环上的点 6 * 6
     *  对方在外环上的点 6 * 6
     *  我方在内环上的点 6 * 6
     *  对方在内环上的点 6 * 6
     *  对方的最新的移动到的点，表示从哪个地方移动点。6 * 6
     *  对方的最新的移动到的点，表示移动到哪个地方。6 * 6
     *  我方是否先手(1的时候，我方先手，否则为0) 6 * 6
     *
     * 总共是 9 * 6 * 6 的局面
     */

    // 公共网络
    torch::nn::Conv2d conv1 { nullptr };
    torch::nn::BatchNorm bat1 { nullptr };
    torch::nn::Conv2d conv2 { nullptr };
    torch::nn::BatchNorm bat2 { nullptr };
    torch::nn::Conv2d conv3 { nullptr };
    torch::nn::BatchNorm bat3 { nullptr };
    torch::nn::Conv2d conv4 { nullptr };
    torch::nn::BatchNorm bat4 { nullptr };

    // 策略网络
    torch::nn::Conv2d pol_conv1 { nullptr };
    torch::nn::BatchNorm pol_bat1 { nullptr };
    torch::nn::Linear pol_fc1 { nullptr };

    // 价值网络
    torch::nn::Conv2d val_conv1 { nullptr };
    torch::nn::BatchNorm val_bat1 { nullptr };
    torch::nn::Linear val_fc1 { nullptr };
    torch::nn::Linear val_fc2 { nullptr };
};

TORCH_MODULE(Net);

struct PolicyValueNet {
    PolicyValueNet();
    ~PolicyValueNet();
    torch::optim::Optimizer* optimizer = nullptr;
    torch::Device device { torch::kCPU };
    Net model;
    std::pair<torch::Tensor, torch::Tensor> policy_value(torch::Tensor);
    std::pair<torch::Tensor, torch::Tensor> train_step(torch::Tensor states_batch, torch::Tensor mcts_probs, torch::Tensor winner_batch);
    void save_model(std::string model_file);
    void load_model(std::string model_file);
};
#endif // POLICY_VALUE_MODEL_H
