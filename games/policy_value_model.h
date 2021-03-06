#pragma once
#include <torch/torch.h>
#include <torch/nn/cloneable.h>


#include <utility>
inline torch::nn::Conv2dOptions conv_options(
    int64_t in_planes, int64_t out_planes, int64_t kerner_size,
    int64_t stride = 1, int64_t padding = 0, bool with_bias = false)
{
    torch::nn::Conv2dOptions conv_options =
        torch::nn::Conv2dOptions(in_planes, out_planes, kerner_size);
    conv_options.stride(stride).padding(padding).bias(with_bias);
    return conv_options;
}
// 残差网络
struct BasicBlockImpl : public torch::nn::Cloneable<BasicBlockImpl>
{
    int64_t stride;
    BasicBlockImpl() = default;
    virtual void reset() override;
    void reset_parameters();
    torch::nn::Conv2d conv1 = nullptr;
    torch::nn::BatchNorm2d bn1 = nullptr;
    torch::nn::Conv2d conv2 = nullptr;
    torch::nn::BatchNorm2d bn2 = nullptr;
    torch::nn::Sequential downsample;
    BasicBlockImpl(int64_t inplanes, int64_t planes, int64_t stride_ = 1,
                   torch::nn::Sequential downsample_ = torch::nn::Sequential());
    torch::Tensor forward(torch::Tensor x);
    private:
        int64_t inplanes;
        int64_t planes;
        int64_t stride_;
        torch::nn::Sequential downsample_;
};

TORCH_MODULE(BasicBlock);

struct NetImpl : public torch::nn::Cloneable<NetImpl>
{
    static const int_fast32_t width  = 6;
    static const int_fast32_t height = 6;
    NetImpl();
    virtual void reset() ;
    std::pair<torch::Tensor, torch::Tensor> forward(torch::Tensor x);

    /*
     * 输入特征:
     *  我方棋局的形态 6 * 6
     *  对方局面的形态 6 * 6
     *  我方在外环上的点 6 * 6
     *  对方在外环上的点 6 * 6
     *  我方在内环上的点 6 * 6
     *  对方在内环上的点 6 * 6
     *  对方的最新的移动到的点，表示移动到哪个地方。6 * 6
     *  是否先手(1的时候，先手，否则为0) 6 * 6
     *
     * 总共是 9 * 6 * 6 的局面
     */

    // 公共网络
    torch::nn::Conv2d conv1{nullptr};
    torch::nn::BatchNorm2d bat1{nullptr};
    torch::nn::Sequential res_layers{nullptr};

    // 策略网络
    torch::nn::Conv2d pol_conv1{nullptr};
    torch::nn::BatchNorm2d pol_bat1{nullptr};
    torch::nn::Linear pol_fc1{nullptr};

    // 价值网络
    torch::nn::Conv2d val_conv1{nullptr};
    torch::nn::BatchNorm2d val_bat1{nullptr};
    torch::nn::Linear val_fc1{nullptr};
    torch::nn::Linear val_fc2{nullptr};
};

TORCH_MODULE(Net);

struct PolicyValueNet
{
    PolicyValueNet();
    PolicyValueNet(int device_ind);
    ~PolicyValueNet();
    torch::optim::Optimizer* optimizer = nullptr;
    Net model;
    std::pair<torch::Tensor, torch::Tensor> policy_value(torch::Tensor);
    std::pair<torch::Tensor, torch::Tensor> train_step(
        torch::Tensor states_batch, torch::Tensor mcts_probs,
        torch::Tensor winner_batch);
    void save_model(std::string model_file);
    void load_model(std::string model_file);
    std::string serialize() const;
    void deserialize(std::string in);
    torch::Device device{torch::kCPU};
};
