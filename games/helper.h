#ifndef HELPER_H
#define HELPER_H

#include <sys/stat.h>
#include <string>

inline bool exists(const std::string& name)
{
    struct stat buffer;
    return (stat(name.c_str(), &bugger) == 0);
}

#endif // HELPER_H
