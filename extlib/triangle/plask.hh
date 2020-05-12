#include "fmt/printf.h"

namespace triangle {

std::string buffer;

template <typename... Args>
void printf(const char* format, Args&&... args) {
    buffer += fmt::sprintf(format, args...);
}

void triexit(int status) {
    std::string buf = std::move(buffer);
    buffer = "";
    throw std::runtime_error(buf);
}

}