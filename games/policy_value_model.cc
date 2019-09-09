#include "policy_value_model.h"
#include <iostream>
#include <torch/torch.h>
#include <utility>

using torch::nn::Conv2dOptions;
NetImpl::NetImpl()
    : conv1(register_module("conv1", torch::nn::Conv2d(Conv2dOptions(9, 256, 3).padding(1))))
    , conv2(register_module("conv2", torch::nn::Conv2d(Conv2dOptions(256, 256, 3).padding(1))))
    , conv3(register_module("conv3", torch::nn::Conv2d(Conv2dOptions(256, 256, 3).padding(1))))
    , conv4(register_module("conv4", torch::nn::Conv2d(Conv2dOptions(256, 256, 3).padding(1))))
    , bat1(register_module("bat1", torch::nn::BatchNorm(torch::nn::BatchNormOptions(256))))
    , bat2(register_module("bat2", torch::nn::BatchNorm(torch::nn::BatchNormOptions(256))))
    , bat3(register_module("bat3", torch::nn::BatchNorm(torch::nn::BatchNormOptions(256))))
    // 策略网络
    , pol_conv1(register_module("pol_conv1", torch::nn::Conv2d(Conv2dOptions(256, 256, 1))))
    , pol_fc1(register_module("pol_fc1", torch::nn::Linear(256 * width * height, width * height * width * height)))
    , pol_bat(register_module("pol_bat", torch::nn::BatchNorm(torch::nn::BatchNormOptions(256))))

    // 价值网络
    , val_conv1(register_module("val_conv1", torch::nn::Conv2d(Conv2dOptions(256, 4, 1))))
    , val_fc1(register_module("val_fc1", torch::nn::Linear(4 * width * height, 1)))

{
}

std::pair<torch::Tensor, torch::Tensor> NetImpl::forward(torch::Tensor x)
{
    // 公共的结构
    x = torch::relu(conv1->forward(x));
    x = torch::relu(bat1(conv2->forward(x)));
    x = torch::relu(bat2(conv3->forward(x)));
    x = torch::relu(bat3(conv4->forward(x)));

    // 策略网络
    auto x_pol = torch::relu(pol_bat(pol_conv1->forward(x)));
    x_pol = x_pol.view({ -1, 256 * height * width });
    x_pol = torch::log_softmax(pol_fc1->forward(x_pol), 1);
    std::cout << x_pol[0].exp_() << std::endl;

    // 价值网络
    auto x_val = torch::relu(val_conv1->forward(x));
    x_val = torch::relu(val_fc1->forward(x_val.view({ -1, 4 * width * height })));
    x_val = torch::tanh(x_val);

    return { x_pol, x_val };
}

PolicyValueNet::PolicyValueNet()
{
    // 宇宙的答案
    torch::DeviceType device_type;
    torch::manual_seed(42);

    if (torch::cuda::is_available()) {
        std::cout << "CUDA available! Training on GPU." << std::endl;
        device_type = torch::kCUDA;
    } else {
        std::cout << "Training on CPU" << std::endl;
        device_type = torch::kCPU;
    }

    device = device_type;

    model->to(device);

    optimizer = new torch::optim::Adam(model->parameters(), torch::optim::AdamOptions(1e-3).weight_decay(1e-4));
}

PolicyValueNet::~PolicyValueNet()
{
    delete optimizer;
}

/*
 * 获得输入一个batch的棋局batsh，获得policy和value
 */
std::pair<torch::Tensor, torch::Tensor> PolicyValueNet::policy_value(const torch::Tensor states)
{
    torch::TensorOptions options;
    torch::Tensor tgt;
    options = options.device(device).dtype(torch::kFloat32);
    tgt = states.to(options);

    model->eval();

    auto result = model->forward(tgt);

    return { result.first.cpu(), result.second.cpu() };
}

/*
 * 进行一次step训练
 */
std::pair<torch::Tensor, torch::Tensor> PolicyValueNet::train_step(torch::Tensor states_batch, torch::Tensor mcts_probs, torch::Tensor winner_batch)
{
    using namespace torch;
    TensorOptions option;
    option = option.device(device);
    option = option.dtype(kFloat32);
    states_batch = states_batch.to(option);
    mcts_probs = mcts_probs.to(option);
    winner_batch = winner_batch.to(option);

    model->train();

    optimizer->zero_grad();

    torch::Tensor log_act_prob, value;
    std::tie(log_act_prob, value) = model->forward(states_batch);

    auto value_loss = mse_loss(value.view({ -1 }), winner_batch);
    auto prob_loss = -mean(sum(mcts_probs.exp() * log_act_prob, 1), kFloat32);
    auto loss = value_loss + prob_loss;

    loss.backward();
    optimizer->step();

    // 计算一次策略的交叉熵，用于性能检测用途
    auto entropy = -mean(sum(log_act_prob * mcts_probs, 1), kFloat32);
    return std::make_pair(loss, entropy);
}

void PolicyValueNet::save_model(std::string model_file)
{
    torch::save(model, model_file);
}

void PolicyValueNet::load_model(std::string model_file)
{
    torch::load(model, model_file);
}
