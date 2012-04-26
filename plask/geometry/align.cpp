#include "align.h"

#include <boost/algorithm/string.hpp>
#include <boost/lexical_cast.hpp>

namespace plask {
namespace align {

namespace details {

Aligner2d<DIRECTION_TRAN>* transAlignerFromString(std::string str) {
    boost::algorithm::to_lower(str);
    if (str == "left" || str == "l") return new Left();
    if (str == "right" || str == "r") return new Right();
    if (str == "center" || str == "c" || str == "m" || str == "middle") return new Center();
    return new Tran(boost::lexical_cast<double>(str));
}

Aligner2d<DIRECTION_LON>* lonAlignerFromString(std::string str) {
    boost::algorithm::to_lower(str);
    if (str == "front" || str == "f") return new Front();
    if (str == "back" || str == "b") return new Back();
    if (str == "center" || str == "c" || str == "loncenter" || str == "m" || str == "middle") return new LonCenter();
    return new Lon(boost::lexical_cast<double>(str));
}

}   // namespace details

Aligner3d<align::DIRECTION_LON, align::DIRECTION_TRAN>* aligner3dFromString(std::string str) {
    boost::algorithm::to_lower(str);
         if (str == "front left" || str == "fl" || str == "left front" || str == "lf") return new FrontLeft();
    else if (str == "center left" || str == "cl" || str == "left center" || str == "lc") return new CenterLeft();
    else if (str == "back left" || str == "bl" || str == "left back" || str == "lb") return new BackLeft();
    else if (str == "front center" || str == "fc" || str == "center front" || str == "lf") return new FrontCenter();
    else if (str == "center center" || str == "cc" || str == "center" || str == "c") return new CenterCenter();
    else if (str == "back center" || str == "bl" || str == "center back" || str == "lb") return new BackCenter();
    else if (str == "front right" || str == "fr" || str == "right front" || str == "rf") return new FrontRight();
    else if (str == "center right" || str == "cr" || str == "right center" || str == "rc") return new CenterRight();
    else if (str == "back right" || str == "br" || str == "right back" || str == "rb") return new BackRight();
    throw BadInput("alignerFromString", "wrong aligner specification");
    return nullptr;
}



}   // namespace align
}   // namespace plask
