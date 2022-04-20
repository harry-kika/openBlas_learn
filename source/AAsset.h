//
// Created by Harry on 2022/4/18.
//

#ifndef OPENBLAS_LEARN_AASSET_H
#define OPENBLAS_LEARN_AASSET_H
#include <fstream>
#include <sys/time.h>
struct AAsset {
    std::ifstream inf;

    AAsset(std::string file) {
        inf = std::ifstream(file, std::ios::binary);
    }

    ~AAsset() {
        inf.close();
    }
};

static int AAsset_read(AAsset* asset, void* buf, size_t count) {
    asset->inf.read(reinterpret_cast<char*>(buf), count);
    return count;
}

static inline int64_t GetTimeInMs()
{
    const int us = 1000;
    struct timeval now;
    gettimeofday(&now, NULL);
    return static_cast<int64_t>(now.tv_sec * us + now.tv_usec / us);
}

#endif //OPENBLAS_LEARN_AASSET_H
