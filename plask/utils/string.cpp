#include "string.h"

namespace plask {

std::tuple< std::string, std::string > splitString2(const std::string& to_split, char splitter) {
    std::string::size_type p = to_split.find(splitter);
    return p == std::string::npos ?
            std::tuple<std::string, std::string>(to_split, "") :
            std::tuple<std::string, std::string>(to_split.substr(0, p), to_split.substr(p+1));

}

    
}       // namespace plask