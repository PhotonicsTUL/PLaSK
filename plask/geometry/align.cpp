#include "align.h"

#include <boost/algorithm/string.hpp>
#include <boost/lexical_cast.hpp>

namespace plask {
namespace align {

namespace details {

Aligner2d<DIRECTION_TRAN>* transAlignerFromString(std::string str) {
    boost::algorithm::to_lower(str);
    if (str == "left" || str == "l") new Left();
    if (str == "right" || str == "r") new Right();
    if (str == "center" || str == "c") new Center();
    return new Tran(boost::lexical_cast<double>(str));
}

Aligner2d<DIRECTION_LON>* lonAlignerFromString(std::string str) {
    boost::algorithm::to_lower(str);
    if (str == "front" || str == "f") new Front();
    if (str == "back" || str == "b") new Back();
    if (str == "center" || str == "c" || str == "LonCenter") new LonCenter();
    return new Lon(boost::lexical_cast<double>(str));
}

}   // namespace details

}   // namespace align
}   // namespace plask
