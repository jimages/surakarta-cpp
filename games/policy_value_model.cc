#include "policy_value_model.h"
#include <iostream>
#include <torch/torch.h>
#include <utility>

using torch::nn::Conv2dOptions;
BasicBlockImpl::BasicBlockImpl(int64_t inplanes, int64_t planes, int64_t stride_,
    torch::nn::Sequential downsample_)
    : conv1(conv_options(inplanes, planes, 3, stride_, 1))
    , bn1(planes)
    , conv2(conv_options(planes, planes, 3, 1, 1))
    , bn2(planes)
    , downsample(downsample_)
{
    register_module("conv1", conv1);
    register_module("bn1", bn1);
    register_module("conv2", conv2);
    register_module("bn2", bn2);
    stride = stride_;
    if (!downsample->is_empty()) {
        register_module("downsample", downsample);
    }
}

torch::Tensor BasicBlockImpl::forward(torch::Tensor x)
{
    at::Tensor residual(x.clone());

    x = conv1->forward(x);
    x = bn1->forward(x);
    x = torch::relu(x);

    x = conv2->forward(x);
    x = bn2->forward(x);

    if (!downsample->is_empty()) {
        residual = downsample->forward(residual);
    }

    x += residual;
    x = torch::relu(x);

    return x;
}
NetImpl::NetImpl()
    : conv1(register_module("conv1", torch::nn::Conv2d(Conv2dOptions(9, 256, 3).padding(1))))
    , bat1(register_module("bat1", torch::nn::BatchNorm(torch::nn::BatchNormOptions(256))))
    , res_layers(torch::nn::Sequential())
    // 策略网络
    , pol_conv1(register_module("pol_conv1", torch::nn::Conv2d(Conv2dOptions(256, 4, 1))))
    , pol_bat1(register_module("pol_bat1", torch::nn::BatchNorm(torch::nn::BatchNormOptions(256))))
    , pol_fc1(register_module("pol_fc1", torch::nn::Linear(4 * width * height, width * height * width * height)))

    // 价值网络
    , val_conv1(register_module("val_conv1", torch::nn::Conv2d(Conv2dOptions(256, 2, 1))))
    , val_bat1(register_module("val_bat1", torch::nn::BatchNorm(torch::nn::BatchNormOptions(2))))
    , val_fc1(register_module("val_fc1", torch::nn::Linear(2 * width * height, 256)))
    , val_fc2(register_module("val_fc2", torch::nn::Linear(256, 1)))

{
    for (int i = 0; i <= 5; ++i) {
        res_layers->push_back(BasicBlock(256, 256));
    }
    register_module("res_layers", res_layers);
}

std::pair<torch::Tensor, torch::Tensor> NetImpl::forward(torch::Tensor x)
{
    // 公共的结构
    x = torch::relu(bat1->forward(conv1->forward(x)));
    x = res_layers->forward(x);

    // 策略网络
    auto x_pol = torch::relu(pol_bat1->forward(pol_conv1->forward(x)));
    x_pol = x_pol.view({ -1, 256 * height * width });
    x_pol = torch::log_softmax(pol_fc1->forward(x_pol), 1);

    // 价值网络
    auto x_val = torch::relu(val_bat1->forward(val_conv1->forward(x)));
    x_val = torch::relu(val_fc1->forward(x_val.view({ -1, 9 * width * height })));
    x_val = torch::tanh(val_fc2->forward(x_val));

    return { x_pol, x_val };
}

PolicyValueNet::PolicyValueNet()
{
    // 宇宙的答案
    torch::manual_seed(42);
    torch::DeviceType device_type;

    if (torch::cuda::is_available()) {
        std::cout << "CUDA available! work on GPU." << std::endl;
        device_type = torch::kCUDA;
    } else {
        std::cout << "work on CPU" << std::endl;
        device_type = torch::kCPU;
    }

    device = torch::Device(device_type);

    model->to(device);

    optimizer = new torch::optim::Adam(model->parameters(), torch::optim::AdamOptions(1e-3).weight_decay(1e-4));
}
PolicyValueNet::PolicyValueNet(int device_ind)
{
    // 宇宙的答案
    torch::manual_seed(42);

    assert(torch::cuda::is_available());
    assert(device_ind < torch::cuda::device_count());
    if (torch::cuda::is_available()) {
        std::cout << "CUDA available! work on GPU." << std::endl;
        device = torch::Device(torch::kCUDA, device_ind);
    }

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
    option = option.device(this->device).dtype(kFloat32);
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

std::string PolicyValueNet::serialize() const
{
    std::stringstream stream;
    torch::save(model, stream);
    return stream.str();
}
void PolicyValueNet::deserialize(std::string in)
{
    std::stringstream stream(in);
    torch::load(model, stream);
    model->to(device);
}
