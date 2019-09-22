#ifndef HELPER_H
#define HELPER_H

#include <sstream>
#include <string>
#include <sys/stat.h>
#include <torch/torch.h>

inline bool exists(const std::string& name)
{
    struct stat buffer;
    return (stat(name.c_str(), &buffer) == 0);
}

template <typename T>
std::string torch_serialize(const T& tensor)
{
    std::stringstream stream;
    torch::save(tensor, stream);
    return stream.str();
}
template <typename T>
std::string model_serialize(const T& tensor)
{
    std::stringstream stream;
    torch::save(tensor, stream);
    return stream.str();
}
template <typename T>
inline T model_deserialize(T model, const std::string& str)
{
    std::stringstream stream(str);
    torch::load(model, stream);
    return model;
}

inline torch::Tensor torch_deserialize(const std::string& str)
{
    std::stringstream stream(str);
    torch::Tensor tens;
    torch::load(tens, stream);
    return tens;
}

#endif // HELPER_H
