#ifndef HELPER_H
#define HELPER_H

#include <string>
#include <sys/stat.h>

inline bool exists(const std::string& name)
{
    struct stat buffer;
    return (stat(name.c_str(), &buffer) == 0);
}

#endif // HELPER_H
