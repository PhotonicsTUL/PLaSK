#include "id.h"

#include <thread>
#include <boost/lexical_cast.hpp>

namespace plask {

std::uint64_t id = 0;
std::mutex id_m;

std::uint64_t getUniqueNumber() {
    std::lock_guard<std::mutex> lock(id_m);
    return ++id;
}

std::string getUniqueString() {
    return boost::lexical_cast<std::string>(getUniqueNumber());
}

}   // namespace plask
