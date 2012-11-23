#include "id.h"

#include <atomic>
#include <boost/lexical_cast.hpp>

namespace plask {

std::atomic<std::uint64_t> id;
//std::uint64_t id;

std::uint64_t getUniqueNumber() {
    return ++id;
}

std::string getUniqueString() {
    return boost::lexical_cast<std::string>(getUniqueNumber());
}

}   // namespace plask
