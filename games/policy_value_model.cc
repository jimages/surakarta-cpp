#include "policy_value_model.h"
#include <iostream>
#include <torch/torch.h>
#include <utility>

using torch::nn::Conv2dOptions;
NetImpl::NetImpl()
{
    conv1 = register_module("conv1", torch::nn::Conv2d(Conv2dOptions(9, 128, 3).padding(1)));
    conv2 = register_module("conv2", torch::nn::Conv2d(Conv2dOptions(128, 256, 3).padding(1)));

    // 策略网络
    pol_conv1 = register_module("pol_conv1", torch::nn::Conv2d(Conv2dOptions(256, 16, 1)));
    pol_fc1 = register_module("pol_fc1", torch::nn::Linear(16 * width * height, width * height * width * height));

    // 价值网络
    val_conv1 = register_module("val_conv1", torch::nn::Conv2d(Conv2dOptions(256, 4, 1)));
    val_fc1 = register_module("val_fc1", torch::nn::Linear(4 * width * height, 64));
    val_fc2 = register_module("val_fc2", torch::nn::Linear(64, 1));
}

std::pair<torch::Tensor, torch::Tensor> NetImpl::forward(torch::Tensor x)
{
    // 公共的结构
    x = torch::relu(conv1->forward(x));
    x = torch::relu(conv2->forward(x));

    // 策略网络
    auto x_pol = torch::relu(pol_conv1->forward(x));
    x_pol = x_pol.view({ -1, 16 * height * width });
    x_pol = torch::log_softmax(pol_fc1->forward(x_pol), 1);

    // 价值网络
    auto x_val = torch::relu(val_conv1->forward(x));
    x_val = torch::relu(val_fc1->forward(x_val.view({ -1, 4 * width * height })));
    x_val = torch::relu(val_fc2->forward(x_val));
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
std::pair<torch::Tensor, torch::Tensor> PolicyValueNet::policy_value(torch::Tensor states)
{
    torch::TensorOptions options;
    options = options.device(device).dtype(torch::kFloat);
    states = states.to(options);

    model->eval();

    auto result = model->forward(states);
    torch::exp_(result.first);

    return result;
}

/*
 * 进行一次step训练
 */
std::pair<torch::Tensor, torch::Tensor> PolicyValueNet::train_step(torch::Tensor states_batch, torch::Tensor mcts_probs, torch::Tensor winner_batch)
{
    using namespace torch;
    TensorOptions option;
    option = option.device(device);
    option = option.dtype(kFloat);
    states_batch = states_batch.to(option);
    mcts_probs = mcts_probs.to(option);
    winner_batch = winner_batch.to(option);

    model->train();

    optimizer->zero_grad();

    torch::Tensor log_act_prob, value;
    std::tie(log_act_prob, value) = model->forward(states_batch);

    auto value_loss = mse_loss(value.view({ -1 }), winner_batch);
    auto prob_loss = -mean(sum(mcts_probs * exp(log_act_prob)), kFloat);
    auto loss = value_loss + prob_loss;

    loss.backward();
    optimizer->step();

    // 计算一次策略的交叉熵，用于性能检测用途
    auto entropy = -mean(sum(exp(log_act_prob) * mcts_probs, kFloat));
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
