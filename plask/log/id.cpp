#include "id.h"

#include <boost/thread.hpp>
#include <boost/lexical_cast.hpp>

namespace plask {

std::uint64_t id = 0;
boost::mutex id_m;

std::uint64_t getUniqueNumber() {
    boost::lock_guard<boost::mutex> lock(id_m);
    return ++id;
}

std::string getUniqueString() {
    return boost::lexical_cast<std::string>(getUniqueNumber());
}

}   // namespace plask
